from ultralytics import YOLO
import torch
import os
import shutil
from pathlib import Path
import yaml

def create_combined_dataset():
    """
    Combine blue and yellow cone datasets into a single multi-class dataset
    """
    # Define paths  
    base_path = Path(r'C:\Users\Joshv\Desktop\QCar')
    blue_dataset = base_path / 'Datasets' / 'blueconesdataset'
    yellow_dataset = base_path / 'Datasets' / 'Yellowconesdataset' 
    combined_dataset = base_path / 'Datasets' / 'combinedataset'
    
    # Print paths for debugging
    print(f"🔍 Looking for datasets:")
    print(f"   Blue: {blue_dataset}")
    print(f"   Yellow: {yellow_dataset}")
    print(f"   Combined output: {combined_dataset}")
    
    # Check if source datasets exist
    if not blue_dataset.exists():
        print(f"❌ Blue dataset not found: {blue_dataset}")
        return None
    if not yellow_dataset.exists():
        print(f"❌ Yellow dataset not found: {yellow_dataset}")
        return None
    
    # Create combined dataset structure
    splits_to_process = []
    
    # Check which splits exist in your datasets
    for split in ['train', 'val', 'test']:
        if (blue_dataset / 'images' / split).exists() or (yellow_dataset / 'images' / split).exists():
            splits_to_process.append(split)
            (combined_dataset / 'images' / split).mkdir(parents=True, exist_ok=True)
            (combined_dataset / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print(f"📋 Processing splits: {splits_to_process}")
    
    print("📁 Created combined dataset structure")
    
    # Count files for progress tracking
    total_blue_images = 0
    total_yellow_images = 0
    
    # Copy and rename files from both datasets
    for split in splits_to_process:
        # Process blue cones
        blue_img_path = blue_dataset / 'images' / split
        blue_label_path = blue_dataset / 'labels' / split
        
        if blue_img_path.exists():
            print(f"📘 Processing blue cones from {split}...")
            for img_file in blue_img_path.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Copy image with blue_ prefix
                    new_img_name = f"blue_{img_file.name}"
                    shutil.copy2(img_file, combined_dataset / 'images' / split / new_img_name)
                    total_blue_images += 1
                    
                    # Copy corresponding label (class 0 = blue_cone)
                    label_file = blue_label_path / f"{img_file.stem}.txt"
                    if label_file.exists():
                        new_label_name = f"blue_{label_file.name}"
                        shutil.copy2(label_file, combined_dataset / 'labels' / split / new_label_name)
                    else:
                        print(f"⚠️ Warning: No label found for {img_file.name}")
        
        # Process yellow cones
        yellow_img_path = yellow_dataset / 'images' / split
        yellow_label_path = yellow_dataset / 'labels' / split
        
        if yellow_img_path.exists():
            print(f"🟡 Processing yellow cones from {split}...")
            for img_file in yellow_img_path.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Copy image with yellow_ prefix
                    new_img_name = f"yellow_{img_file.name}"
                    shutil.copy2(img_file, combined_dataset / 'images' / split / new_img_name)
                    total_yellow_images += 1
                    
                    # Copy and modify label (class 1 = yellow_cone)
                    label_file = yellow_label_path / f"{img_file.stem}.txt"
                    if label_file.exists():
                        new_label_name = f"yellow_{label_file.name}"
                        new_label_path = combined_dataset / 'labels' / split / new_label_name
                        
                        # Read original label and change class to 1
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                        
                        with open(new_label_path, 'w') as f:
                            for line in lines:
                                parts = line.strip().split()
                                if parts:
                                    # Change class from 0 to 1 for yellow cones
                                    parts[0] = '1'
                                    f.write(' '.join(parts) + '\n')
                    else:
                        print(f"⚠️ Warning: No label found for {img_file.name}")
    
    # Create dataset.yaml file
    yaml_splits = {}
    for split in splits_to_process:
        yaml_splits[split] = f'images/{split}'
    
    dataset_yaml = {
        'path': str(combined_dataset.absolute()),
        **yaml_splits,  # Add train, val, test paths dynamically
        'nc': 2,  # Number of classes
        'names': ['blue_cone', 'yellow_cone']
    }
    
    yaml_file_path = combined_dataset / 'dataset.yaml'
    with open(yaml_file_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print("✅ Combined dataset created successfully!")
    print(f"📍 Dataset path: {combined_dataset}")
    print(f"📊 Summary:")
    print(f"   🔵 Blue cone images: {total_blue_images}")
    print(f"   🟡 Yellow cone images: {total_yellow_images}")
    print(f"   📝 Total images: {total_blue_images + total_yellow_images}")
    print(f"📄 Created dataset.yaml at: {yaml_file_path}")
    return combined_dataset

def train_combined_cone_detector():
    """
    Train YOLO model to detect both blue and yellow cones
    """
    # Create combined dataset first
    dataset_path = create_combined_dataset()
    
    if dataset_path is None:
        print("❌ Cannot proceed with training - dataset creation failed")
        return None
    
    # Check if dataset.yaml exists
    yaml_file = dataset_path / 'dataset.yaml'
    if not yaml_file.exists():
        print(f"❌ dataset.yaml not found at: {yaml_file}")
        return None
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')  # nano version - faster training
    # For better accuracy, use: YOLO('yolov8s.pt') or YOLO('yolov8m.pt')
    
    print(f"📋 Training with dataset: {yaml_file}")
    
    # Train the model
    results = model.train(
        data=str(yaml_file),
        epochs=30,          # Number of training epochs
        imgsz=640,          # Image size
        batch=8,           # Batch size (reduce if you get memory errors)
        device=device,      # Use GPU if available
        patience=10,        # Early stopping patience
        save=True,          # Save checkpoints
        plots=True,         # Generate training plots
        verbose=True,       # Print training progress
        name='combined_cones'  # Custom run name
    )
    
    print("🎉 Training completed!")
    print(f"📁 Model saved to: runs/detect/combined_cones/weights/best.pt")
    
    return results

def test_combined_model():
    """
    Test the trained combined model
    """
    # Load the trained model
    model = YOLO('runs/detect/combined_cones/weights/best.pt')
    
    # Test on validation set
    results = model.val()
    
    print(f"📊 Validation Results:")
    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")
    
    # Test on both blue and yellow validation images
    combined_dataset = Path(r'C:\Users\Joshv\Desktop\QCar\Datasets\combinedataset')
    
    if (combined_dataset / 'images' / 'val').exists():
        test_results = model.predict(
            source=str(combined_dataset / 'images' / 'val'),
            save=True,
            conf=0.5,  # Confidence threshold
            name='combined_test'
        )
        print("🔍 Predictions saved to: runs/detect/combined_test/")
    
    return results

def predict_on_new_image(image_path):
    """
    Use trained model to predict on a new image
    """
    model = YOLO('runs/detect/combined_cones/weights/best.pt')
    
    results = model.predict(
        source=image_path,
        save=True,
        conf=0.5,
        show_labels=True,
        show_conf=True
    )
    
    # Print detections
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                print(f"Detected: {class_name} (confidence: {confidence:.2f})")
    
    return results

if __name__ == "__main__":
    print("🚀 Starting YOLO training for combined cone detection...")
    
    # Train the combined model
    train_results = train_combined_cone_detector()
    
    if train_results is not None:
        # Test the model
        print("\n🧪 Testing trained model...")
        test_results = test_combined_model()
        
        print("\n✅ All done! Check the 'runs' folder for results and trained model.")
        print("📋 Model can now detect both blue_cone (class 0) and yellow_cone (class 1)")
        
        # Example usage for prediction
        # predict_on_new_image('path_to_your_test_image.jpg')
    else:
        print("❌ Training failed - please check your dataset folders and try again.")