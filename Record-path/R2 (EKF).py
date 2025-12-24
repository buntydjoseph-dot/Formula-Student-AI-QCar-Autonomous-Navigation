import os
import numpy as np
import time
import cv2
import torch
from ultralytics import YOLO
import json
import math

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.real_time import QLabsRealTime
from qvl.system import QLabsSystem
from qvl.traffic_cone import QLabsTrafficCone
import pal.resources.rtmodels as rtmodels

from pal.products.qcar import QCarCameras, QCar

class AdvancedQcarEKF:
    def __init__(self, x0, P0, Q, R_gps, R_visual, R_imu):
        """
        Advanced EKF with multiple sensor fusion
        
        Args:
            x0: Initial state [x, y, theta, vx, vy, omega]
            P0: Initial covariance matrix
            Q: Process noise covariance
            R_gps: GPS measurement noise covariance
            R_visual: Visual odometry measurement noise covariance
            R_imu: IMU measurement noise covariance
        """
        self.L = 0.257  # Wheelbase
        self.dt_integration = 0.01  # Fine timestep for integration
        
        # Extended state: [x, y, theta, vx, vy, omega]
        self.xHat = x0.copy()
        self.P = P0.copy()
        self.Q = Q
        self.R_gps = R_gps
        self.R_visual = R_visual 
        self.R_imu = R_imu
        self.I = np.eye(len(x0))
        
        # Store previous measurements for velocity estimation
        self.prev_gps_pos = None
        self.prev_gps_time = None
        self.prev_visual_pos = None
        self.prev_visual_time = None

    def f(self, X, u, dt):
        """
        Nonlinear state transition function
        State: [x, y, theta, vx, vy, omega]
        Input: [speed, steering_angle]
        """
        x, y, theta, vx, vy, omega = X.flatten()
        speed, delta = u
        
        # Bicycle model with velocity states
        # Update velocities first
        vx_new = speed * np.cos(delta)  # Longitudinal velocity
        vy_new = speed * np.sin(delta)  # Lateral velocity (simplified)
        omega_new = (speed / self.L) * np.tan(delta)  # Angular velocity
        
        # Integrate position and orientation
        x_new = x + (vx * np.cos(theta) - vy * np.sin(theta)) * dt
        y_new = y + (vx * np.sin(theta) + vy * np.cos(theta)) * dt
        theta_new = theta + omega * dt
        
        return np.array([[x_new], [y_new], [theta_new], [vx_new], [vy_new], [omega_new]])

    def Jf(self, X, u, dt):
        """
        Jacobian of state transition function
        """
        x, y, theta, vx, vy, omega = X.flatten()
        speed, delta = u
        
        # Jacobian matrix (6x6)
        Jf = np.array([
            [1, 0, -(vx * np.sin(theta) + vy * np.cos(theta)) * dt, np.cos(theta) * dt, -np.sin(theta) * dt, 0],
            [0, 1, (vx * np.cos(theta) - vy * np.sin(theta)) * dt, np.sin(theta) * dt, np.cos(theta) * dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 0.9, 0, 0],  # Velocity damping
            [0, 0, 0, 0, 0.9, 0],
            [0, 0, 0, 0, 0, 0.95]  # Angular velocity damping
        ])
        return Jf

    def prediction(self, dt, u):
        """EKF Prediction step with improved integration"""
        # Multiple integration steps for better accuracy
        num_steps = max(1, int(dt / self.dt_integration))
        dt_step = dt / num_steps
        
        for _ in range(num_steps):
            self.xHat = self.f(self.xHat, u, dt_step)
        
        # Update covariance
        F = self.Jf(self.xHat, u, dt)
        self.P = F @ self.P @ F.T + self.Q * dt
        
        # Wrap angle
        self.xHat[2, 0] = self.wrap_to_pi(self.xHat[2, 0])

    def correction_gps(self, gps_measurement, current_time):
        """
        EKF correction step using GPS data
        gps_measurement: [x, y] position from GPS/QLabs
        """
        if gps_measurement is None:
            return
            
        # Measurement model: H maps state to GPS observation
        H_gps = np.array([
            [1, 0, 0, 0, 0, 0],  # x position
            [0, 1, 0, 0, 0, 0],  # y position
        ])
        
        # Predicted measurement
        h_pred = H_gps @ self.xHat
        
        # Innovation
        y_innovation = np.array([[gps_measurement[0]], [gps_measurement[1]]]) - h_pred
        
        # Innovation covariance
        S = H_gps @ self.P @ H_gps.T + self.R_gps
        
        # Kalman gain
        K = self.P @ H_gps.T @ np.linalg.inv(S)
        
        # State update
        self.xHat = self.xHat + K @ y_innovation
        
        # Covariance update
        self.P = (self.I - K @ H_gps) @ self.P
        
        # Wrap angle
        self.xHat[2, 0] = self.wrap_to_pi(self.xHat[2, 0])
        
        # Estimate GPS-derived velocity if we have previous measurement
        if self.prev_gps_pos is not None and self.prev_gps_time is not None:
            dt_gps = current_time - self.prev_gps_time
            if dt_gps > 0.1:  # Only if sufficient time has passed
                vx_gps = (gps_measurement[0] - self.prev_gps_pos[0]) / dt_gps
                vy_gps = (gps_measurement[1] - self.prev_gps_pos[1]) / dt_gps
                
                # Update velocity estimates with GPS-derived velocities
                self.correction_velocity([vx_gps, vy_gps], 'gps')
        
        self.prev_gps_pos = gps_measurement.copy()
        self.prev_gps_time = current_time

    def correction_visual_odometry(self, visual_delta, confidence):
        """
        EKF correction using visual odometry from cone tracking
        visual_delta: [dx, dy, dtheta] relative motion estimate
        confidence: confidence in visual measurement (0-1)
        """
        if visual_delta is None or confidence < 0.3:
            return
            
        # Adaptive measurement noise based on confidence
        R_visual_adaptive = self.R_visual / confidence
        
        # Measurement model for relative motion
        H_visual = np.array([
            [1, 0, 0, 0, 0, 0],  # x displacement
            [0, 1, 0, 0, 0, 0],  # y displacement  
            [0, 0, 1, 0, 0, 0],  # theta change
        ])
        
        # Convert relative motion to absolute position
        current_pred = H_visual @ self.xHat
        expected_pos = current_pred + np.array([[visual_delta[0]], [visual_delta[1]], [visual_delta[2]]])
        
        # Innovation
        y_innovation = expected_pos - current_pred
        y_innovation[2, 0] = self.wrap_to_pi(y_innovation[2, 0])
        
        # Innovation covariance
        S = H_visual @ self.P @ H_visual.T + R_visual_adaptive
        
        # Kalman gain
        K = self.P @ H_visual.T @ np.linalg.inv(S)
        
        # State update
        self.xHat = self.xHat + K @ y_innovation
        
        # Covariance update
        self.P = (self.I - K @ H_visual) @ self.P
        
        # Wrap angle
        self.xHat[2, 0] = self.wrap_to_pi(self.xHat[2, 0])

    def correction_imu(self, imu_data):
        """
        EKF correction using IMU data (simulated from steering/speed)
        imu_data: [ax, ay, omega_z] accelerations and angular velocity
        """
        if imu_data is None:
            return
            
        # Measurement model for IMU
        H_imu = np.array([
            [0, 0, 0, 0, 0, 1],  # Angular velocity measurement
        ])
        
        # Innovation (only angular velocity for now)
        h_pred = H_imu @ self.xHat
        y_innovation = np.array([[imu_data[2]]]) - h_pred
        
        # Innovation covariance
        S = H_imu @ self.P @ H_imu.T + self.R_imu
        
        # Kalman gain
        K = self.P @ H_imu.T @ np.linalg.inv(S)
        
        # State update
        self.xHat = self.xHat + K @ y_innovation
        
        # Covariance update  
        self.P = (self.I - K @ H_imu) @ self.P

    def correction_velocity(self, velocity_measurement, source='encoder'):
        """
        Correct velocity states using encoder or GPS-derived velocities
        velocity_measurement: [vx, vy]
        """
        H_vel = np.array([
            [0, 0, 0, 1, 0, 0],  # vx
            [0, 0, 0, 0, 1, 0],  # vy
        ])
        
        # Select appropriate noise model
        R_vel = self.R_gps if source == 'gps' else np.eye(2) * 0.1
        
        # Innovation
        h_pred = H_vel @ self.xHat
        y_innovation = np.array([[velocity_measurement[0]], [velocity_measurement[1]]]) - h_pred
        
        # Innovation covariance
        S = H_vel @ self.P @ H_vel.T + R_vel
        
        # Kalman gain
        K = self.P @ H_vel.T @ np.linalg.inv(S)
        
        # Update
        self.xHat = self.xHat + K @ y_innovation
        self.P = (self.I - K @ H_vel) @ self.P

    def wrap_to_pi(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def get_state(self):
        """Return current state estimate"""
        return {
            'position': [self.xHat[0, 0], self.xHat[1, 0]],
            'heading': self.xHat[2, 0],
            'velocity': [self.xHat[3, 0], self.xHat[4, 0]],
            'angular_velocity': self.xHat[5, 0],
            'covariance': self.P
        }

class TrueEKFNavigationWithLapRecording:
    def __init__(self, model_path, path_file=None, save_path=r"C:\Users\Joshv\Desktop\Presentation\Code\Spawn&Navigation\Path\EKFRecorded"):
        self.model = YOLO(model_path)
        
        self.qcar = QCar()
        self.cameras = QCarCameras(enableBack=False, enableFront=True, enableLeft=False, enableRight=False)
        
        # Path saving configuration
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        
        # Control parameters
        self.max_speed = 0.1
        self.min_speed = 0.02
        self.target_speed = 0.06
        self.max_steering = 0.6
        
        # YOLO parameters
        self.yellow_class_id = 1
        self.blue_class_id = 0
        self.confidence_threshold = 0.35
        self.front_cam_idx = 3
        
        # Image parameters
        self.img_width = 820
        self.img_height = 410
        self.img_center_x = self.img_width // 2
        
        # Control parameters
        self.steering_invert = -1
        self.previous_steering = 0.0
        self.steering_alpha = 0.8
        
        # Navigation state
        self.running = False
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.last_detection_time = time.time()
        
        # Initialize advanced EKF
        x0 = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])  # [x, y, theta, vx, vy, omega]
        P0 = np.eye(6) * 0.1
        Q = np.diagflat([0.01, 0.01, 0.005, 0.05, 0.05, 0.02])  # Process noise
        R_gps = np.eye(2) * 0.1      # GPS noise
        R_visual = np.eye(3) * 0.2   # Visual odometry noise  
        R_imu = np.eye(1) * 0.05     # IMU noise
        
        self.ekf = AdvancedQcarEKF(x0, P0, Q, R_gps, R_visual, R_imu)
        
        # QLabs interface for GPS-like position
        self.qlabs_qcar = None
        
        # Path following
        self.learned_path = []
        self.path_following_enabled = False
        
        # Visual odometry tracking
        self.prev_lane_center = None
        self.prev_cone_positions = []
        
        # LAP RECORDING VARIABLES (from second code)
        self.start_position = None
        self.start_zone = None
        self.current_lap = 1
        self.path_data = []
        self.lap_completed = False
        self.frame_count = 0
        
        if path_file:
            self.load_path_data(path_file)

    def initialize_qlabs_qcar(self, qlabs):
        """Initialize QLabs QCar for position sensing"""
        self.qlabs_qcar = QLabsQCar(qlabs)
        self.qlabs_qcar.actorNumber = 0
        print("QLabs QCar initialized for GPS-like positioning and lap recording")

    def get_gps_measurement(self):
        """Get GPS-like position measurement from QLabs"""
        if self.qlabs_qcar:
            try:
                success, location, rotation, frontHit, rearHit = self.qlabs_qcar.set_velocity_and_request_state(
                    0.0, 0.0, False, False, False, False, False
                )
                if success:
                    return [float(location[0]), float(location[1])], float(rotation[2])
            except:
                pass
        return None, None

    def record_path_point(self, steering, speed, current_state):
        """
        Record path point with both QLabs position and EKF state data
        """
        # Get QLabs position
        location, rotation = self.get_gps_measurement()
        if location is None:
            return
        
        # Define start/finish zone (adjust coordinates based on your track)
        if self.start_position is None:
            self.start_position = [float(location[0]), float(location[1]), float(location[2]) if len(location) > 2 else 0.0]
            # Define start/finish zone around starting position
            self.start_zone = {
                'x_min': self.start_position[0] - 2.0,
                'x_max': self.start_position[0] + 2.0, 
                'y_min': self.start_position[1] - 2.0,
                'y_max': self.start_position[1] + 2.0
            }
            print(f"Start position: [{self.start_position[0]:.2f}, {self.start_position[1]:.2f}]")
            print(f"Start zone: X[{self.start_zone['x_min']:.1f}, {self.start_zone['x_max']:.1f}] Y[{self.start_zone['y_min']:.1f}, {self.start_zone['y_max']:.1f}]")
            print(f"Saving EKF lap data to: {self.save_path}")
        
        # Add comprehensive path data including EKF state
        path_point = {
            # QLabs ground truth
            'qlabs_location': [float(location[0]), float(location[1]), float(location[2]) if len(location) > 2 else 0.0],
            'qlabs_rotation': [float(rotation[0]) if hasattr(rotation, '__len__') else 0.0, 
                              float(rotation[1]) if hasattr(rotation, '__len__') and len(rotation) > 1 else 0.0, 
                              float(rotation[2]) if hasattr(rotation, '__len__') and len(rotation) > 2 else float(rotation)],
            
            # EKF estimated state
            'ekf_position': [float(current_state['position'][0]), float(current_state['position'][1])],
            'ekf_heading': float(current_state['heading']),
            'ekf_velocity': [float(current_state['velocity'][0]), float(current_state['velocity'][1])],
            'ekf_angular_velocity': float(current_state['angular_velocity']),
            
            # Position uncertainty from EKF covariance
            'position_uncertainty': float(np.sqrt(current_state['covariance'][0,0] + current_state['covariance'][1,1])),
            'heading_uncertainty': float(np.sqrt(current_state['covariance'][2,2])),
            
            # Control commands
            'steering': float(steering),
            'speed': float(speed),
            'timestamp': float(time.time()),
            
            # Additional metadata
            'lap': int(self.current_lap),
            'frame_count': int(self.frame_count)
        }
        
        self.path_data.append(path_point)
        
        print(f"QLabs: [{location[0]:.2f}, {location[1]:.2f}] | EKF: [{current_state['position'][0]:.2f}, {current_state['position'][1]:.2f}] | Points: {len(self.path_data)}")
        
        # Check if in start/finish zone after moving away from start
        if len(self.path_data) > 100:  # Must have moved away first
            in_start_zone = (self.start_zone['x_min'] <= location[0] <= self.start_zone['x_max'] and 
                           self.start_zone['y_min'] <= location[1] <= self.start_zone['y_max'])
            
            if in_start_zone and not self.lap_completed:
                self.lap_completed = True
                distance_to_start = math.sqrt(
                    (float(location[0]) - self.start_position[0])**2 + 
                    (float(location[1]) - self.start_position[1])**2
                )
                print(f"LAP {self.current_lap} COMPLETED! Back in start zone. Distance: {distance_to_start:.2f}m")
                
                # Save comprehensive lap data
                try:
                    filename = f"ekf_lap_{self.current_lap}_data.json"
                    filepath = os.path.join(self.save_path, filename)
                    
                    # Create summary statistics
                    lap_summary = {
                        'lap_number': self.current_lap,
                        'total_points': len(self.path_data),
                        'start_time': self.path_data[0]['timestamp'] if self.path_data else 0,
                        'end_time': self.path_data[-1]['timestamp'] if self.path_data else 0,
                        'lap_duration': self.path_data[-1]['timestamp'] - self.path_data[0]['timestamp'] if len(self.path_data) > 1 else 0,
                        'avg_speed': np.mean([p['speed'] for p in self.path_data]),
                        'max_speed': np.max([p['speed'] for p in self.path_data]),
                        'avg_position_uncertainty': np.mean([p['position_uncertainty'] for p in self.path_data]),
                        'max_position_uncertainty': np.max([p['position_uncertainty'] for p in self.path_data])
                    }
                    
                    complete_data = {
                        'lap_summary': lap_summary,
                        'path_points': self.path_data
                    }
                    
                    with open(filepath, 'w') as f:
                        json.dump(complete_data, f, indent=2)
                    
                    print(f"EKF lap data saved to: {filepath}")
                    print(f"Lap duration: {lap_summary['lap_duration']:.1f}s | Avg uncertainty: {lap_summary['avg_position_uncertainty']:.3f}m")
                    
                except Exception as e:
                    print(f"Error saving EKF lap data: {e}")
                
                self.current_lap += 1
            
            elif not in_start_zone and self.lap_completed:
                # Reset flag only when car leaves the start zone
                self.lap_completed = False
                print("Exited start zone - ready for next lap")

    def estimate_visual_odometry(self, yellow_cones, blue_cones, dt):
        """
        Estimate relative motion using visual features (cone positions)
        Returns: [dx, dy, dtheta], confidence
        """
        if not yellow_cones and not blue_cones:
            return None, 0.0
        
        current_cone_positions = []
        
        # Extract cone positions
        for cone in yellow_cones + blue_cones:
            current_cone_positions.append([cone['center'][0], cone['center'][1], 'yellow' if cone in yellow_cones else 'blue'])
        
        if len(self.prev_cone_positions) == 0:
            self.prev_cone_positions = current_cone_positions
            return None, 0.0
        
        # Simple visual odometry based on cone movement
        if len(current_cone_positions) >= 2 and len(self.prev_cone_positions) >= 2:
            # Match cones and estimate motion
            dx_visual = 0.0
            dy_visual = 0.0
            dtheta_visual = 0.0
            
            # Simplified: assume average cone movement represents vehicle motion
            for i, current_cone in enumerate(current_cone_positions[:2]):  # Use first 2 cones
                if i < len(self.prev_cone_positions):
                    prev_cone = self.prev_cone_positions[i]
                    # Reverse the cone movement to get vehicle movement
                    dx_visual += (prev_cone[0] - current_cone[0]) * 0.01  # Scale pixel to meters
                    dy_visual += (prev_cone[1] - current_cone[1]) * 0.01
            
            dx_visual /= min(len(current_cone_positions), 2)
            dy_visual /= min(len(current_cone_positions), 2)
            
            confidence = min(len(current_cone_positions) / 4.0, 1.0)  # More cones = higher confidence
            
            self.prev_cone_positions = current_cone_positions
            return [dx_visual, dy_visual, dtheta_visual], confidence
        
        self.prev_cone_positions = current_cone_positions
        return None, 0.0

    def simulate_imu_data(self, speed, steering, dt):
        """
        Simulate IMU data based on vehicle dynamics
        Returns: [ax, ay, omega_z]
        """
        # Simplified IMU simulation
        omega_z = (speed / self.ekf.L) * np.tan(steering)
        ax = speed / dt if dt > 0 else 0.0  # Longitudinal acceleration
        ay = omega_z * speed  # Centripetal acceleration
        
        # Add some noise to simulate real IMU
        noise_scale = 0.1
        ax += np.random.normal(0, noise_scale)
        ay += np.random.normal(0, noise_scale)  
        omega_z += np.random.normal(0, noise_scale * 0.1)
        
        return [ax, ay, omega_z]

    def load_path_data(self, filename):
        """Load learned path data"""
        try:
            with open(filename, 'r') as f:
                self.learned_path = json.load(f)
            self.path_following_enabled = True
            print(f"Loaded path with {len(self.learned_path)} waypoints")
            return True
        except Exception as e:
            print(f"Failed to load path: {e}")
            return False

    def get_path_guidance(self, current_state):
        """Get guidance from learned path"""
        if not self.path_following_enabled or not self.learned_path:
            return 0.0, self.target_speed, 0.0
        
        current_pos = current_state['position']
        
        # Find closest path point
        min_distance = float('inf')
        closest_index = 0
        
        for i, point in enumerate(self.learned_path):
            if 'location' in point:
                path_pos = point['location']
                distance = math.sqrt(
                    (current_pos[0] - path_pos[0])**2 + 
                    (current_pos[1] - path_pos[1])**2
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_index = i
        
        # Lookahead
        lookahead = min(15, len(self.learned_path) - closest_index - 1)
        target_index = closest_index + lookahead
        
        if target_index < len(self.learned_path):
            target_point = self.learned_path[target_index]
            path_steering = target_point.get('steering', 0.0)
            path_speed = target_point.get('speed', self.target_speed) * 1.3  # More aggressive
            path_confidence = max(0.0, 1.0 - (min_distance / 3.0))
            return path_steering, path_speed, path_confidence
        
        return 0.0, self.target_speed, 0.0

    def detect_cones_in_frame(self, frame):
        """Detect cones using YOLO"""
        if frame is None:
            return [], []
            
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        yellow_cones = []
        blue_cones = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                
                cone_data = {
                    'bbox': [x1, y1, x2, y2],
                    'center': [center_x, center_y],
                    'confidence': confidence,
                    'area': area
                }
                
                if class_id == self.yellow_class_id:
                    yellow_cones.append(cone_data)
                elif class_id == self.blue_class_id:
                    blue_cones.append(cone_data)
        
        return yellow_cones, blue_cones

    def calculate_control_commands(self, yellow_cones, blue_cones, current_state, dt):
        """Calculate control commands using EKF state and sensor fusion"""
        
        # Vision-based steering
        vision_steering, vision_confidence = self.calculate_vision_steering(yellow_cones, blue_cones)
        
        # Path-based guidance
        path_steering, path_speed, path_confidence = self.get_path_guidance(current_state)
        
        # Intelligent fusion based on confidence and EKF uncertainty
        pos_uncertainty = np.sqrt(current_state['covariance'][0,0] + current_state['covariance'][1,1])
        
        if vision_confidence > 0.6 and pos_uncertainty < 1.0:
            # High vision confidence and good localization
            final_steering = 0.8 * vision_steering + 0.2 * path_steering
            final_speed = self.target_speed
        elif path_confidence > 0.5 and pos_uncertainty < 2.0:
            # Good path following and reasonable localization
            final_steering = 0.3 * vision_steering + 0.7 * path_steering
            final_speed = path_speed
        elif vision_confidence > 0.3:
            # Fall back to vision
            final_steering = vision_steering
            final_speed = self.target_speed * 0.8
        else:
            # Conservative fallback
            final_steering = 0.0
            final_speed = self.min_speed
        
        # Apply constraints
        final_steering = np.clip(final_steering, -self.max_steering, self.max_steering)
        final_speed = np.clip(final_speed, 0.0, self.max_speed)
        
        return final_steering, final_speed, vision_confidence, path_confidence

    def calculate_vision_steering(self, yellow_cones, blue_cones):
        """Calculate steering based on cone detection"""
        lane_center = None
        confidence = 0.0
        
        yellow_target = max(yellow_cones, key=lambda x: x['area']) if yellow_cones else None
        blue_target = max(blue_cones, key=lambda x: x['area']) if blue_cones else None
        
        if yellow_target and blue_target:
            yellow_x = yellow_target['center'][0]
            blue_x = blue_target['center'][0]
            lane_center = (yellow_x + blue_x) / 2
            confidence = 0.9
        elif blue_target:
            blue_x = blue_target['center'][0]
            lane_center = blue_x + 120
            confidence = 0.6
        elif yellow_target:
            yellow_x = yellow_target['center'][0]
            lane_center = yellow_x - 120
            confidence = 0.6
        
        if lane_center is not None:
            error = lane_center - self.img_center_x
            raw_steering = np.clip(error / 200.0, -1.0, 1.0) * self.steering_invert
            steering = self.steering_alpha * raw_steering + (1 - self.steering_alpha) * self.previous_steering
            self.previous_steering = steering
            return steering, confidence
        
        return 0.0, 0.0

    def draw_detections(self, frame, yellow_cones, blue_cones, current_state, vision_conf, path_conf):
        """Enhanced visualization with EKF state and lap recording info"""
        if frame is None:
            return None
            
        frame_copy = frame.copy()
        
        # Draw cone detections
        for cone in yellow_cones:
            x1, y1, x2, y2 = map(int, cone['bbox'])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame_copy, f"Y {cone['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        for cone in blue_cones:
            x1, y1, x2, y2 = map(int, cone['bbox'])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame_copy, f"B {cone['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw center line
        cv2.line(frame_copy, (self.img_center_x, 0), (self.img_center_x, self.img_height), 
                (255, 255, 255), 2)
        
        # Enhanced state display
        pos = current_state['position']
        vel = current_state['velocity']
        heading = current_state['heading']
        
        cv2.putText(frame_copy, f"EKF Pos: [{pos[0]:.2f}, {pos[1]:.2f}]", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame_copy, f"EKF Vel: [{vel[0]:.2f}, {vel[1]:.2f}]", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame_copy, f"Heading: {np.degrees(heading):.1f}°", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Lap recording info
        cv2.putText(frame_copy, f"LAP: {self.current_lap}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_copy, f"Path points: {len(self.path_data)}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Confidence display
        cv2.putText(frame_copy, f"Vision: {vision_conf:.2f} | Path: {path_conf:.2f}", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Uncertainty indicator
        pos_uncertainty = np.sqrt(current_state['covariance'][0,0] + current_state['covariance'][1,1])
        uncertainty_color = (0, 255, 0) if pos_uncertainty < 1.0 else (0, 255, 255) if pos_uncertainty < 2.0 else (0, 0, 255)
        cv2.putText(frame_copy, f"Pos Uncertainty: {pos_uncertainty:.2f}m", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, uncertainty_color, 2)
        
        # Control commands
        cv2.putText(frame_copy, f"Speed: {self.current_speed:.3f} | Steering: {self.current_steering:.3f}", 
                   (10, self.img_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame_copy

    def run_navigation(self, qlabs, show_video=True, max_runtime=600):
        """Main navigation loop with true EKF sensor fusion and lap recording"""
        self.initialize_qlabs_qcar(qlabs)
        self.running = True
        start_time = time.time()
        last_time = time.time()
        
        try:
            with self.qcar:
                with self.cameras:
                    while self.running and (time.time() - start_time) < max_runtime:
                        current_time = time.time()
                        dt = current_time - last_time
                        last_time = current_time
                        
                        if dt <= 0:
                            continue
                        
                        self.cameras.readAll()
                        front_frame = self.cameras.csi[self.front_cam_idx].imageData.copy() if self.cameras.csi[self.front_cam_idx] is not None else None
                        
                        if front_frame is not None:
                            # 1. EKF Prediction
                            control_input = [self.current_speed * 10.0, self.current_steering]
                            self.ekf.prediction(dt, control_input)
                            
                            # 2. Sensor measurements and corrections
                            
                            # GPS correction
                            gps_pos, gps_heading = self.get_gps_measurement()
                            if gps_pos is not None:
                                self.ekf.correction_gps(gps_pos, current_time)
                            
                            # Detect cones
                            yellow_cones, blue_cones = self.detect_cones_in_frame(front_frame)
                            
                            # Visual odometry correction
                            visual_delta, visual_conf = self.estimate_visual_odometry(yellow_cones, blue_cones, dt)
                            if visual_delta is not None:
                                self.ekf.correction_visual_odometry(visual_delta, visual_conf)
                            
                            # IMU correction (simulated)
                            imu_data = self.simulate_imu_data(self.current_speed, self.current_steering, dt)
                            self.ekf.correction_imu(imu_data)
                            
                            # 3. Get current state estimate
                            current_state = self.ekf.get_state()
                            
                            # 4. Calculate control commands using fused state
                            steering, speed, vision_conf, path_conf = self.calculate_control_commands(
                                yellow_cones, blue_cones, current_state, dt)
                            
                            # 5. Apply control commands
                            self.qcar.read_write_std(throttle=speed, steering=steering)
                            self.current_speed = speed
                            self.current_steering = steering
                            
                            # 6. Record lap data every few frames (similar to second code)
                            if self.frame_count % 5 == 0:
                                self.record_path_point(steering, speed, current_state)
                            
                            # 7. Update detection timing
                            total_cones = len(yellow_cones) + len(blue_cones)
                            if total_cones > 0:
                                self.last_detection_time = time.time()
                            elif time.time() - self.last_detection_time > 5.0:
                                self.qcar.read_write_std(throttle=0.0, steering=0.0)
                                time.sleep(1)
                                self.last_detection_time = time.time()
                            
                            # 8. Display visualization
                            if show_video:
                                display_frame = self.draw_detections(front_frame, yellow_cones, blue_cones, 
                                                                   current_state, vision_conf, path_conf)
                                cv2.imshow('EKF Navigation with Lap Recording', display_frame)
                                
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord('q'):
                                    break
                                elif key == ord('s'):
                                    self.qcar.read_write_std(throttle=0.0, steering=0.0)
                                    time.sleep(2)
                                elif key == ord('r'):  # Reset EKF
                                    print("Resetting EKF state...")
                                    x0 = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
                                    P0 = np.eye(6) * 0.1
                                    self.ekf.xHat = x0
                                    self.ekf.P = P0
                            
                            self.frame_count += 1
                        
                        time.sleep(0.05)  # 20 Hz control loop
                
                # Clean shutdown
                self.qcar.read_write_std(throttle=0.0, steering=0.0)
                
        except KeyboardInterrupt:
            print("Navigation interrupted by user")
        except Exception as e:
            print(f"Navigation error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                self.qcar.read_write_std(throttle=0.0, steering=0.0)
            except:
                pass
            
            if show_video:
                cv2.destroyAllWindows()
            
            self.running = False
            print("Navigation system shutdown complete")

def setup_qlabs_connection():
    """Setup QLabs connection"""
    qlabs = QuanserInteractiveLabs()
    qlabs.open("localhost")
    
    hSystem = QLabsSystem(qlabs)
    hSystem.set_title_string('EKF Navigation with Lap Recording')
    
    return qlabs

def main():
    """Main function"""
    MODEL_PATH = r"C:\Users\Joshv\Desktop\Presentation\Dataset\combinedataset\runs\combined_cones\weights\best.pt"
    PATH_FILE = r"C:\Users\Joshv\Desktop\Presentation\Code\Spawn&Navigation\Path\TrackOne\lap_4_path.json"
    SAVE_PATH = r"C:\Users\Joshv\Desktop\Presentation\Code\Spawn&Navigation\Path\EKFRecorded"
    
    print("=== EKF Navigation with Lap Recording ===")
    print("Features:")
    print("- Extended Kalman Filter with 6-DOF state estimation")
    print("- GPS-like position corrections from QLabs")
    print("- Visual odometry from cone tracking")
    print("- Simulated IMU integration")
    print("- Velocity state estimation and correction")
    print("- Intelligent sensor fusion with confidence weighting")
    print("- Path following with learned trajectory guidance")
    print("- Comprehensive lap data recording (QLabs + EKF)")
    print("- Position uncertainty tracking")
    print("\nControls:")
    print("- 'q': Quit navigation")
    print("- 's': Emergency stop (2 seconds)")
    print("- 'r': Reset EKF state to origin")
    print(f"- Lap data saves to: {SAVE_PATH}")
    print("=" * 50)
    
    try:
        qlabs = setup_qlabs_connection()
        time.sleep(2)
        
        navigator = TrueEKFNavigationWithLapRecording(MODEL_PATH, PATH_FILE, SAVE_PATH)
        navigator.run_navigation(qlabs, show_video=True, max_runtime=600)
        
    except KeyboardInterrupt:
        print("System shutdown requested")
    except Exception as e:
        print(f"System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("System shutdown complete")

if __name__ == '__main__':
    main() 
    
    def draw_detections(self, frame, yellow_cones, blue_cones, current_state, vision_conf, path_conf):
        """Enhanced visualization with EKF state and lap recording info"""
        if frame is None:
            return None
            
        frame_copy = frame.copy()
        
        # Draw cone detections
        for cone in yellow_cones:
            x1, y1, x2, y2 = map(int, cone['bbox'])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame_copy, f"Y {cone['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        for cone in blue_cones:
            x1, y1, x2, y2 = map(int, cone['bbox'])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame_copy, f"B {cone['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw center line
        cv2.line(frame_copy, (self.img_center_x, 0), (self.img_center_x, self.img_height), 
                (255, 255, 255), 2)
        
        # Enhanced state display
        pos = current_state['position']
        vel = current_state['velocity']
        heading = current_state['heading']
        
        cv2.putText(frame_copy, f"EKF Pos: [{pos[0]:.2f}, {pos[1]:.2f}]", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame_copy, f"EKF Vel: [{vel[0]:.2f}, {vel[1]:.2f}]", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame_copy, f"Heading: {np.degrees(heading):.1f}°", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Lap recording info
        cv2.putText(frame_copy, f"LAP: {self.current_lap}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_copy, f"Path points: {len(self.path_data)}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Confidence display
        cv2.putText(frame_copy, f"Vision: {vision_conf:.2f} | Path: {path_conf:.2f}", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Uncertainty indicator
        pos_uncertainty = np.sqrt(current_state['covariance'][0,0] + current_state['covariance'][1,1])
        uncertainty_color = (0, 255, 0) if pos_uncertainty < 1.0 else (0, 255, 255) if pos_uncertainty < 2.0 else (0, 0, 255)
        cv2.putText(frame_copy, f"Pos Uncertainty: {pos_uncertainty:.2f}m", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, uncertainty_color, 2)
        
        # Control commands
        cv2.putText(frame_copy, f"Speed: {self.current_speed:.3f} | Steering: {self.current_steering:.3f}", 
                   (10, self.img_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame_copy

    def run_navigation(self, qlabs, show_video=True, max_runtime=600):
        """Main navigation loop with true EKF sensor fusion and lap recording"""
        self.initialize_qlabs_qcar(qlabs)
        self.running = True
        start_time = time.time()
        last_time = time.time()
        
        try:
            with self.qcar:
                with self.cameras:
                    while self.running and (time.time() - start_time) < max_runtime:
                        current_time = time.time()
                        dt = current_time - last_time
                        last_time = current_time
                        
                        if dt <= 0:
                            continue
                        
                        self.cameras.readAll()
                        front_frame = self.cameras.csi[self.front_cam_idx].imageData.copy() if self.cameras.csi[self.front_cam_idx] is not None else None
                        
                        if front_frame is not None:
                            # 1. EKF Prediction
                            control_input = [self.current_speed * 10.0, self.current_steering]
                            self.ekf.prediction(dt, control_input)
                            
                            # 2. Sensor measurements and corrections
                            
                            # GPS correction
                            gps_pos, gps_heading = self.get_gps_measurement()
                            if gps_pos is not None:
                                self.ekf.correction_gps(gps_pos, current_time)
                            
                            # Detect cones
                            yellow_cones, blue_cones = self.detect_cones_in_frame(front_frame)
                            
                            # Visual odometry correction
                            visual_delta, visual_conf = self.estimate_visual_odometry(yellow_cones, blue_cones, dt)
                            if visual_delta is not None:
                                self.ekf.correction_visual_odometry(visual_delta, visual_conf)
                            
                            # IMU correction (simulated)
                            imu_data = self.simulate_imu_data(self.current_speed, self.current_steering, dt)
                            self.ekf.correction_imu(imu_data)
                            
                            # 3. Get current state estimate
                            current_state = self.ekf.get_state()
                            
                            # 4. Calculate control commands using fused state
                            steering, speed, vision_conf, path_conf = self.calculate_control_commands(
                                yellow_cones, blue_cones, current_state, dt)
                            
                            # 5. Apply control commands
                            self.qcar.read_write_std(throttle=speed, steering=steering)
                            self.current_speed = speed
                            self.current_steering = steering
                            
                            # 6. Record lap data every few frames (similar to second code)
                            if self.frame_count % 5 == 0:
                                self.record_path_point(steering, speed, current_state)
                            
                            # 7. Update detection timing
                            total_cones = len(yellow_cones) + len(blue_cones)
                            if total_cones > 0:
                                self.last_detection_time = time.time()
                            elif time.time() - self.last_detection_time > 5.0:
                                self.qcar.read_write_std(throttle=0.0, steering=0.0)
                                time.sleep(1)
                                self.last_detection_time = time.time()
                            
                            # 8. Display visualization
                            if show_video:
                                display_frame = self.draw_detections(front_frame, yellow_cones, blue_cones, 
                                                                   current_state, vision_conf, path_conf)
                                cv2.imshow('EKF Navigation with Lap Recording', display_frame)
                                
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord('q'):
                                    break
                                elif key == ord('s'):
                                    self.qcar.read_write_std(throttle=0.0, steering=0.0)
                                    time.sleep(2)
                                elif key == ord('r'):  # Reset EKF
                                    print("Resetting EKF state...")
                                    x0 = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
                                    P0 = np.eye(6) * 0.1
                                    self.ekf.xHat = x0
                                    self.ekf.P = P0
                            
                            self.frame_count += 1
                        
                        time.sleep(0.05)  # 20 Hz control loop
                
                # Clean shutdown
                self.qcar.read_write_std(throttle=0.0, steering=0.0)
                
        except KeyboardInterrupt:
            print("Navigation interrupted by user")
        except Exception as e:
            print(f"Navigation error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                self.qcar.read_write_std(throttle=0.0, steering=0.0)
            except:
                pass
            
            if show_video:
                cv2.destroyAllWindows()
            
            self.running = False
            print("Navigation system shutdown complete")

def setup_qlabs_connection():
    """Setup QLabs connection"""
    qlabs = QuanserInteractiveLabs()
    qlabs.open("localhost")
    
    hSystem = QLabsSystem(qlabs)
    hSystem.set_title_string('EKF Navigation with Lap Recording')
    
    return qlabs

def main():
    """Main function"""
    MODEL_PATH = r"C:\Users\Joshv\Desktop\Presentation\Dataset\combinedataset\runs\combined_cones\weights\best.pt"
    PATH_FILE = r"C:\Users\Joshv\Desktop\Presentation\Code\Spawn&Navigation\Path\TrackOne\lap_4_path.json"
    SAVE_PATH = r"C:\Users\Joshv\Desktop\Presentation\Code\Spawn&Navigation\Path\EKFRecorded"
    
    print("=== EKF Navigation with Lap Recording ===")
    print("Features:")
    print("- Extended Kalman Filter with 6-DOF state estimation")
    print("- GPS-like position corrections from QLabs")
    print("- Visual odometry from cone tracking")
    print("- Simulated IMU integration")
    print("- Velocity state estimation and correction")
    print("- Intelligent sensor fusion with confidence weighting")
    print("- Path following with learned trajectory guidance")
    print("- Comprehensive lap data recording (QLabs + EKF)")
    print("- Position uncertainty tracking")
    print("\nControls:")
    print("- 'q': Quit navigation")
    print("- 's': Emergency stop (2 seconds)")
    print("- 'r': Reset EKF state to origin")
    print(f"- Lap data saves to: {SAVE_PATH}")
    print("=" * 50)
    
    try:
        qlabs = setup_qlabs_connection()
        time.sleep(2)
        
        navigator = TrueEKFNavigationWithLapRecording(MODEL_PATH, PATH_FILE, SAVE_PATH)
        navigator.run_navigation(qlabs, show_video=True, max_runtime=600)
        
    except KeyboardInterrupt:
        print("System shutdown requested")
    except Exception as e:
        print(f"System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("System shutdown complete")

if __name__ == '__main__':
    main()