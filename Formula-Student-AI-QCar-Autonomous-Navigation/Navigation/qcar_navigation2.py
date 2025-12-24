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

class QcarEKF:
    def __init__(self, x0, P0, Q):
        self.L = 0.257
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.I = np.eye(3)

    def f(self, X, u, dt):
        x, y, theta = X[0,0], X[1,0], X[2,0]
        speed, delta = u[0], u[1]
        
        x_new = x + speed * np.cos(theta) * dt
        y_new = y + speed * np.sin(theta) * dt
        theta_new = theta + (speed / self.L) * np.tan(delta) * dt
        
        return np.array([[x_new], [y_new], [theta_new]])

    def Jf(self, X, u, dt):
        x, y, theta = X[0,0], X[1,0], X[2,0]
        speed, delta = u[0], u[1]
        
        Jf = np.array([
            [1, 0, -speed * np.sin(theta) * dt],
            [0, 1,  speed * np.cos(theta) * dt],
            [0, 0,  1]
        ])
        return Jf

    def prediction(self, dt, u):
        self.xHat = self.f(self.xHat, u, dt)
        F = self.Jf(self.xHat, u, dt)
        self.P = F @ self.P @ F.T + self.Q
        self.xHat[2,0] = self.wrap_to_pi(self.xHat[2,0])

    def correction(self, y, R):
        H = np.eye(3)
        innovation = y - H @ self.xHat
        innovation[2,0] = self.wrap_to_pi(innovation[2,0])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.xHat = self.xHat + K @ innovation
        self.P = (self.I - K @ H) @ self.P
        self.xHat[2,0] = self.wrap_to_pi(self.xHat[2,0])

    def wrap_to_pi(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

class EKFPathNavigation:
    def __init__(self, model_path, path_file=None):
        self.model = YOLO(model_path)
        
        self.qcar = QCar()
        self.cameras = QCarCameras(enableBack=False, enableFront=True, enableLeft=False, enableRight=False)
        
        self.max_speed = 0.08
        self.min_speed = 0.02
        self.target_speed = 0.05
        self.max_steering = 0.6
        
        self.yellow_class_id = 1
        self.blue_class_id = 0
        self.confidence_threshold = 0.35
        
        self.front_cam_idx = 3
        
        self.img_width = 820
        self.img_height = 410
        self.img_center_x = self.img_width // 2
        
        self.estimated_lane_width = 260
        self.inner_boundary_offset = 150
        self.outer_boundary_offset = 150
        self.racing_line_bias = 0.5
        
        self.steering_invert = -1
        self.previous_steering = 0.0
        self.steering_alpha = 0.8
        
        self.running = False
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.last_detection_time = time.time()
        
        x0 = np.array([[0.0], [0.0], [0.0]])
        P0 = np.eye(3) * 0.1
        Q = np.diagflat([0.01, 0.01, 0.01])
        self.ekf = QcarEKF(x0, P0, Q)
        
        self.learned_path = []
        self.path_following_enabled = False
        
        if path_file:
            self.load_path_data(path_file)

    def load_path_data(self, filename):
        try:
            with open(filename, 'r') as f:
                self.learned_path = json.load(f)
            self.path_following_enabled = True
            return True
        except:
            return False

    def get_current_position_ekf(self):
        return [self.ekf.xHat[0,0], self.ekf.xHat[1,0], self.ekf.xHat[2,0]]

    def find_closest_path_point(self, current_pos):
        if not self.learned_path or not current_pos:
            return 0
            
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
        
        return closest_index

    def get_path_guidance(self, current_pos):
        if not self.path_following_enabled or not self.learned_path or not current_pos:
            return 0.0, self.target_speed, 0.0
        
        closest_index = self.find_closest_path_point(current_pos)
        lookahead = min(10, len(self.learned_path) - closest_index - 1)
        target_index = closest_index + lookahead
        
        if target_index < len(self.learned_path):
            target_point = self.learned_path[target_index]
            path_steering = target_point.get('steering', 0.0)
            path_speed = target_point.get('speed', self.target_speed)
            path_pos = target_point['location']
            distance_to_path = math.sqrt(
                (current_pos[0] - path_pos[0])**2 + 
                (current_pos[1] - path_pos[1])**2
            )
            path_confidence = max(0.0, 1.0 - (distance_to_path / 5.0))
            return path_steering, path_speed * 1.2, path_confidence
        
        return 0.0, self.target_speed, 0.0

    def detect_cones_in_frame(self, frame):
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

    def calculate_simple_steering(self, yellow_cones, blue_cones, dt):
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
        else:
            steering = 0.0
        
        speed_input = self.current_speed * 10.0
        steering_input = self.current_steering
        self.ekf.prediction(dt, [speed_input, steering_input])
        
        current_pos = self.get_current_position_ekf()
        path_steering, path_speed, path_confidence = self.get_path_guidance(current_pos)
        
        if path_confidence > 0.5 and confidence > 0.5:
            final_steering = 0.7 * steering + 0.3 * path_steering
            final_speed = self.target_speed
        elif confidence > 0.3:
            final_steering = steering
            final_speed = self.target_speed
        elif path_confidence > 0.3:
            final_steering = path_steering
            final_speed = path_speed
        else:
            final_steering = 0.0
            final_speed = self.min_speed
        
        all_cones = yellow_cones + blue_cones
        if all_cones:
            max_area = max(cone['area'] for cone in all_cones)
            if max_area > 15000:
                final_speed = self.min_speed
            elif max_area > 8000:
                final_speed *= 0.7
        
        final_steering = np.clip(final_steering, -self.max_steering, self.max_steering)
        final_speed = np.clip(final_speed, 0.0, self.max_speed)
        
        return final_steering, final_speed, confidence, path_confidence

    def draw_detections(self, frame, yellow_cones, blue_cones, vision_conf, path_conf):
        if frame is None:
            return None
            
        frame_copy = frame.copy()
        
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
        
        cv2.line(frame_copy, (self.img_center_x, 0), (self.img_center_x, self.img_height), 
                (255, 255, 255), 2)
        
        pos = self.get_current_position_ekf()
        cv2.putText(frame_copy, f"EKF Pos: [{pos[0]:.1f}, {pos[1]:.1f}]", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        conf_text = f"Vision: {vision_conf:.2f} | Path: {path_conf:.2f}"
        cv2.putText(frame_copy, conf_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        status_text = f"Speed: {self.current_speed:.3f} | Steering: {self.current_steering:.2f}"
        cv2.putText(frame_copy, status_text, (10, self.img_height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame_copy

    def run_navigation(self, show_video=True, max_runtime=600):
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
                        
                        self.cameras.readAll()
                        
                        front_frame = self.cameras.csi[self.front_cam_idx].imageData.copy() if self.cameras.csi[self.front_cam_idx] is not None else None
                        
                        if front_frame is not None:
                            yellow_cones, blue_cones = self.detect_cones_in_frame(front_frame)
                            steering, speed, vision_conf, path_conf = self.calculate_simple_steering(yellow_cones, blue_cones, dt)
                            
                            self.qcar.read_write_std(throttle=speed, steering=steering)
                            
                            self.current_speed = speed
                            self.current_steering = steering
                            
                            total_cones = len(yellow_cones) + len(blue_cones)
                            if total_cones > 0:
                                self.last_detection_time = time.time()
                            elif time.time() - self.last_detection_time > 8.0:
                                self.qcar.read_write_std(throttle=0.0, steering=0.0)
                                time.sleep(1)
                                self.last_detection_time = time.time()
                            
                            if show_video:
                                display_frame = self.draw_detections(front_frame, yellow_cones, blue_cones, vision_conf, path_conf)
                                cv2.imshow('Simple Steering Navigation', display_frame)
                                
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord('q'):
                                    break
                                elif key == ord('s'):
                                    self.qcar.read_write_std(throttle=0.0, steering=0.0)
                                    time.sleep(2)
                        
                        time.sleep(0.05)
                
                self.qcar.read_write_std(throttle=0.0, steering=0.0)
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Error: {e}")
        finally:
            try:
                self.qcar.read_write_std(throttle=0.0, steering=0.0)
            except:
                pass
            
            if show_video:
                cv2.destroyAllWindows()
            
            self.running = False

def setup_qlabs_connection():
    qlabs = QuanserInteractiveLabs()
    qlabs.open("localhost")
    
    hSystem = QLabsSystem(qlabs)
    hSystem.set_title_string('Simple Direct Steering Navigation')
    
    return qlabs

def main():
    MODEL_PATH = r"C:\Users\Joshv\Desktop\Presentation\Dataset\combinedataset\runs\combined_cones\weights\best.pt"
    PATH_FILE = r"C:\Users\Joshv\Desktop\Presentation\Code\Spawn&Navigation\Path\Trackthree\straight_path_1.json"
    
    try:
        qlabs = setup_qlabs_connection()
        time.sleep(2)
        
        navigator = EKFPathNavigation(MODEL_PATH, PATH_FILE)
        navigator.run_navigation(show_video=True, max_runtime=600)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()