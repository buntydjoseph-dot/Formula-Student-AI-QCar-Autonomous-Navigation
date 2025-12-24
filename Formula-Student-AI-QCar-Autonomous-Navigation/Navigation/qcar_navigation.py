import os
import numpy as np
import time
import cv2
import torch
from ultralytics import YOLO
import threading
from datetime import datetime

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.real_time import QLabsRealTime
from qvl.system import QLabsSystem
from qvl.traffic_cone import QLabsTrafficCone
import pal.resources.rtmodels as rtmodels

from pal.products.qcar import QCarCameras, QCar

class QCarSingleCameraNavigation:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        self.qcar = QCar()
        self.cameras = QCarCameras(enableBack=False, enableFront=True, enableLeft=False, enableRight=False)
        
        self.max_speed = 0.06     
        self.min_speed = 0.02     
        self.target_speed = 0.04  
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
        self.track_boundary_margin = 50
        
        self.max_cone_distance = 15000
        self.blue_cone_priority_multiplier = -1.0
        self.min_cone_area = 2000
        
        self.lane_center_memory = []
        self.memory_size = 10
        
        self.steering_filter_alpha = 0.7
        self.previous_steering = 0.0
        self.steering_invert = -1
        
        self.running = False
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.last_detection_time = time.time()
    
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
    
    def calculate_lane_center(self, yellow_cones, blue_cones):
        lane_center = None
        confidence = 0.0
        
        yellow_target = None
        blue_target = None
        
        if yellow_cones:
            yellow_target = max(yellow_cones, key=lambda x: x['area'])
        if blue_cones:
            blue_target = max(blue_cones, key=lambda x: x['area'])
        
        if yellow_target and blue_target:
            yellow_x = yellow_target['center'][0]
            blue_x = blue_target['center'][0]
            
            lane_center = blue_x + (yellow_x - blue_x) * self.racing_line_bias
            
            avg_area = (yellow_target['area'] + blue_target['area']) / 2
            separation = abs(yellow_x - blue_x)
            confidence = min(avg_area / 5000, 1.0) * min(separation / 200, 1.0)
            
        elif blue_target:
            blue_x = blue_target['center'][0]
            safe_distance_from_inner = self.inner_boundary_offset * 0.8
            lane_center = blue_x + safe_distance_from_inner
            confidence = 0.7
            
        elif yellow_target:
            yellow_x = yellow_target['center'][0]
            safety_margin_from_outer = self.outer_boundary_offset * 1.5
            lane_center = yellow_x - safety_margin_from_outer
            confidence = 0.6
        
        if lane_center is not None:
            margin = 50
            lane_center = np.clip(lane_center, margin, self.img_width - margin)
        
        return lane_center, confidence
    
    def calculate_steering_command(self, yellow_cones, blue_cones):
        lane_center, confidence = self.calculate_lane_center(yellow_cones, blue_cones)
        
        if lane_center is not None:
            self.lane_center_memory.append(lane_center)
            if len(self.lane_center_memory) > self.memory_size:
                self.lane_center_memory.pop(0)
            
            smoothed_center = np.mean(self.lane_center_memory)
            
            error = smoothed_center - self.img_center_x
            raw_steering = np.clip(error / (self.img_width / 2), -1.0, 1.0)
            
            raw_steering = raw_steering * self.steering_invert
            
            steering_angle = (self.steering_filter_alpha * raw_steering + 
                            (1 - self.steering_filter_alpha) * self.previous_steering)
            self.previous_steering = steering_angle
            
        else:
            steering_angle = 0.0
        
        all_cones = yellow_cones + blue_cones
        if all_cones:
            max_area = max(cone['area'] for cone in all_cones)
            if max_area > 15000:
                speed = self.min_speed
            elif max_area > 8000:
                speed = self.min_speed * 1.5
            else:
                speed = self.target_speed
        else:
            speed = self.min_speed
        
        steering_angle = np.clip(steering_angle, -self.max_steering, self.max_steering)
        speed = np.clip(speed, 0.0, self.max_speed)
        
        return steering_angle, speed
    
    def draw_detections(self, frame, yellow_cones, blue_cones):
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
        
        cv2.putText(frame_copy, "FRONT CAMERA", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        status_text = f"Speed: {self.current_speed:.3f} | Steering: {self.current_steering:.2f}"
        cv2.putText(frame_copy, status_text, (10, self.img_height-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        count_text = f"Yellow: {len(yellow_cones)}, Blue: {len(blue_cones)}"
        cv2.putText(frame_copy, count_text, (10, self.img_height-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_copy
    
    def run_navigation(self, show_video=True, max_runtime=600):
        self.running = True
        start_time = time.time()
        frame_count = 0
        
        try:
            with self.qcar:
                with self.cameras:
                    while self.running and (time.time() - start_time) < max_runtime:
                        self.cameras.readAll()
                        
                        front_frame = self.cameras.csi[self.front_cam_idx].imageData.copy() if self.cameras.csi[self.front_cam_idx] is not None else None
                        
                        if front_frame is not None:
                            yellow_cones, blue_cones = self.detect_cones_in_frame(front_frame)
                            
                            steering, speed = self.calculate_steering_command(yellow_cones, blue_cones)
                            
                            speed = min(speed, self.max_speed)
                            
                            self.qcar.read_write_std(throttle=speed, steering=steering)
                            
                            self.current_speed = speed
                            self.current_steering = steering
                            
                            total_cones = len(yellow_cones) + len(blue_cones)
                            if total_cones > 0:
                                self.last_detection_time = time.time()
                            
                            elif time.time() - self.last_detection_time > 3.0:
                                self.qcar.read_write_std(throttle=0.0, steering=0.0)
                                time.sleep(1)
                                self.last_detection_time = time.time()
                            
                            if show_video:
                                display_frame = self.draw_detections(front_frame, yellow_cones, blue_cones)
                                cv2.imshow('QCar Front Camera Navigation', display_frame)
                                
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord('q'):
                                    break
                                elif key == ord('s'):
                                    self.qcar.read_write_std(throttle=0.0, steering=0.0)
                                    time.sleep(2)
                            
                            frame_count += 1
                        
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
    hSystem.set_title_string('QCar Single Front Camera Autonomous Cone Navigation')
    
    return qlabs

def main():
    MODEL_PATH = r"C:\Users\Joshv\Desktop\Presentation\Dataset\combinedataset\runs\combined_cones\weights\best.pt"
    
    try:
        qlabs = setup_qlabs_connection()
        time.sleep(2)
        
        navigator = QCarSingleCameraNavigation(MODEL_PATH)
        navigator.run_navigation(show_video=True, max_runtime=600)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()