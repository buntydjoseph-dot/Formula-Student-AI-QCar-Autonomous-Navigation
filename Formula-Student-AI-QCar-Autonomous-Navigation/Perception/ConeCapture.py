# region: package imports
import os
import numpy as np
import time
import cv2
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.real_time import QLabsRealTime
from qvl.system import QLabsSystem
from qvl.traffic_cone import QLabsTrafficCone
import pal.resources.rtmodels as rtmodels
from pal.products.qcar import QCarCameras, QCar
import threading
from datetime import datetime
#endregion

def setup_basic(initialPosition=[0.504, 0.002, 0], initialOrientation=[0, 0, -np.pi/2]):
    """Basic setup - exactly like your original"""
    qlabs = QuanserInteractiveLabs()
    qlabs.open("localhost")
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()

    hqcar = QLabsQCar(qlabs)
    hqcar.spawn_id(actorNumber=0, location=initialPosition, 
                   rotation=initialOrientation, waitForConfirmation=True)
    hqcar.possess()
    QLabsRealTime().start_real_time_model(rtmodels.QCAR)
    
    hSystem = QLabsSystem(qlabs)
    hSystem.set_title_string('Cone Data Capture - Different Angles')
    
    return qlabs, hqcar

def reset_car_position(qlabs, initialPosition=[0.504, 0.002, 0], initialOrientation=[0, 0, -np.pi/2]):
    """Reset car to center position between layouts"""
    print("🔄 Resetting car to center position...")
    # Destroy and respawn the car at center position
    hqcar = QLabsQCar(qlabs)
    hqcar.destroy()
    time.sleep(0.5)
    hqcar.spawn_id(actorNumber=0, location=initialPosition, 
                   rotation=initialOrientation, waitForConfirmation=True)
    hqcar.possess()
    time.sleep(1)  # Wait for position to update
    return hqcar

def create_angle_layouts():
    """Create layouts for different viewing angles - only patterns 1, 2, 3"""
    layouts = {}
    
    # Layout 1: Close circle - for close-up detailed shots
    layouts['close_detail'] = []
    radius = 3.0
    for i in range(8):
        angle = 2 * np.pi * i / 8
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        layouts['close_detail'].append([x, y, 0])
    
    # Layout 2: Medium distance circle
    layouts['medium_distance'] = []
    radius = 4.5
    for i in range(8):
        angle = 2 * np.pi * i / 8
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        layouts['medium_distance'].append([x, y, 0])
    
    # Layout 3: Wide spread for far shots
    layouts['wide_spread'] = []
    radius = 6
    for i in range(10):
        angle = 2 * np.pi * i / 10
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        layouts['wide_spread'].append([x, y, 0])
    
    return layouts

def spawn_basic_cones(qlabs, positions, layout_name=""):
    """Spawn cones - basic version"""
    print(f"🟡 Spawning {layout_name}: {len(positions)} cones")
    
    # Clear existing cones
    for i in range(1, 25):
        try:
            cone = QLabsTrafficCone(qlabs)
            cone.destroy_actor(i)
        except:
            pass
    
    # Spawn cones
    for i, pos in enumerate(positions, start=1):
        cone = QLabsTrafficCone(qlabs)
        cone.spawn_id(actorNumber=i, location=pos, rotation=[0, 0, 0], 
                     scale=[1, 1, 1], configuration=2, waitForConfirmation=True)
        cone.set_material_properties(materialSlot=0, color=[1,1,0], roughness=1, metallic=False)
        cone.set_material_properties(materialSlot=1, color=[0, 0, 0])

def drive_for_angles(duration=45.0, pattern="circle"):
    """Drive patterns to capture different angles - only patterns 1, 2, 3"""
    car = QCar()
    with car:
        print(f"🚗 Driving pattern: {pattern} for {duration}s")
        start_time = time.time()
        
        if pattern == "tight_circle":
            # Very tight, slow circles for close-up shots
            while time.time() - start_time < duration:
                car.read_write_std(0.02, 0.9)
                time.sleep(0.05)
                
        elif pattern == "medium_circle":
            # Medium circles
            while time.time() - start_time < duration:
                car.read_write_std(0.04, 0.6)
                time.sleep(0.05)
                
        elif pattern == "wide_circle":
            # Wide circles for distance shots
            while time.time() - start_time < duration:
                car.read_write_std(0.06, 0.4)
                time.sleep(0.05)
        
        car.read_write_std(0.0, 0.0)
        print(f"✅ {pattern} complete")

def capture_only_camera_loop(save_folder, run_time, layout_name):
    """Camera capture without any model - pure image saving"""
    layout_folder = os.path.join(save_folder, layout_name)
    os.makedirs(layout_folder, exist_ok=True)
    
    cameras = QCarCameras(enableBack=True, enableFront=True, enableLeft=True, enableRight=True)
    print(f"📸 Capturing {layout_name} for {run_time}s...")

    with cameras:
        start_time = time.time()
        frame_idx = 0
        saved_count = 0

        while time.time() - start_time < run_time:
            cameras.readAll()

            for i, c in enumerate(cameras.csi):
                if c is not None:
                    image = c.imageData.copy()
                    
                    # Save raw image with high quality
                    filename = f"{layout_folder}/camera_{i}_frame_{frame_idx:05d}.jpg"
                    cv2.imwrite(filename, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_count += 1

            frame_idx += 1
            
            # Show progress every 50 frames
            if frame_idx % 50 == 0:
                elapsed = time.time() - start_time
                remaining = run_time - elapsed
                print(f"    📸 {saved_count} images saved, {remaining:.1f}s remaining...")
            
            time.sleep(0.08)  # ~12 FPS

    print(f"✅ {layout_name} complete: {saved_count} images saved")
    return saved_count

def main():
    try:
        print("🏁 Setting up QLabs...")
        qlabs_conn, hqcar = setup_basic()

        print("🎯 Starting capture session - NO MODEL REQUIRED")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_folder = f"raw_cone_dataset_{timestamp}"
        os.makedirs(main_folder, exist_ok=True)
        
        layouts = create_angle_layouts()
        
        # Define driving patterns for each layout - only 3 patterns now
        driving_patterns = {
            'close_detail': 'tight_circle',
            'medium_distance': 'medium_circle', 
            'wide_spread': 'wide_circle'
        }
        
        capture_time = 4.0  # 10 seconds per layout
        total_images = 0
        
        print(f"📋 Will capture {len(layouts)} layouts, {capture_time}s each")
        print(f"📂 Saving to: {main_folder}")
        
        for layout_name, positions in layouts.items():
            print(f"\n📍 Layout {list(layouts.keys()).index(layout_name) + 1}/{len(layouts)}: {layout_name}")
            
            # Reset car to center position before each layout
            hqcar = reset_car_position(qlabs_conn)
            
            # Spawn cones
            spawn_basic_cones(qlabs_conn, positions, layout_name)
            time.sleep(2)  # Wait for cones to spawn
            
            # Get driving pattern
            pattern = driving_patterns[layout_name]
            
            # Start driving
            driving_thread = threading.Thread(
                target=drive_for_angles, 
                kwargs={'duration': capture_time, 'pattern': pattern}
            )
            driving_thread.start()
            
            # Capture images (NO MODEL)
            layout_images = capture_only_camera_loop(main_folder, capture_time, layout_name)
            total_images += layout_images
            
            driving_thread.join()
            print(f"✅ Completed: {layout_name}")
        
        print(f"\n🎉 RAW CAPTURE COMPLETE!")
        print(f"📊 Total images captured: {total_images}")
        print(f"📁 Data saved to: {main_folder}")
        print(f"🔄 Average images per layout: {total_images/len(layouts):.0f}")
        
        # Show folder structure
        print(f"\n📂 Folder structure:")
        for layout_name in layouts.keys():
            layout_path = os.path.join(main_folder, layout_name)
            if os.path.exists(layout_path):
                image_count = len([f for f in os.listdir(layout_path) if f.endswith('.jpg')])
                print(f"   📁 {layout_name}: {image_count} images")

    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user.")
    except Exception as e:
        print(f"⚠️ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("🧹 Done.")

if __name__ == '__main__':
    main()