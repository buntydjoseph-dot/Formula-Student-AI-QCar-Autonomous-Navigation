import cv2
import os
from pathlib import Path

class SimpleYOLOAnnotator:
    def __init__(self, image_dir, labels_dir, cone_type="Orangecone"):
        self.image_dir = Path(image_dir)
        self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(exist_ok=True)
        self.cone_type = cone_type
        
        self.current_image = None
        self.current_image_path = None
        # Support multiple image formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(list(self.image_dir.glob(ext)))
        
        self.current_index = 0
        self.drawing = False
        self.start_point = None
        self.boxes = []
        
        print(f"Found {len(self.image_files)} images in {self.image_dir}")
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                end_point = (x, y)
                self.add_box(self.start_point, end_point)
                self.drawing = False
                self.start_point = None
                self.draw_image()
    
    def add_box(self, start, end):
        # Convert to YOLO format (normalized coordinates)
        img_h, img_w = self.current_image.shape[:2]
        
        x1, y1 = start
        x2, y2 = end
        
        # Ensure x1,y1 is top-left and x2,y2 is bottom-right
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Skip very small boxes
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            print("Box too small, skipped")
            return
        
        # Convert to YOLO format: center_x, center_y, width, height (normalized)
        center_x = ((x1 + x2) / 2) / img_w
        center_y = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        
        # Class 0 for both cone types (will be converted later when combining datasets)
        self.boxes.append([0, center_x, center_y, width, height])
        print(f"Added box {len(self.boxes)}: center({center_x:.3f}, {center_y:.3f}) size({width:.3f}, {height:.3f})")
    
    def draw_image(self):
        # Create a copy to draw on
        display_img = self.current_image.copy()
        img_h, img_w = display_img.shape[:2]
        
        # Choose color based on cone type
        color = (0, 255, 255) if self.cone_type == "Orangecone" else (255, 0, 0)  
        
        # Draw existing boxes
        for i, box in enumerate(self.boxes):
            _, center_x, center_y, width, height = box
            
            # Convert back to pixel coordinates
            x1 = int((center_x - width/2) * img_w)
            y1 = int((center_y - height/2) * img_h)
            x2 = int((center_x + width/2) * img_w)
            y2 = int((center_y + height/2) * img_h)
            
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_img, f"{self.cone_type} #{i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add instructions on image
        cv2.putText(display_img, f"Image {self.current_index + 1}/{len(self.image_files)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f"Boxes: {len(self.boxes)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        window_title = f'YOLO Annotator - Click and drag around {self.cone_type}s'
        cv2.imshow(window_title, display_img)
    
    def save_labels(self):
        if not self.boxes:
            # Delete label file if no boxes
            label_file = self.labels_dir / f"{self.current_image_path.stem}.txt"
            if label_file.exists():
                label_file.unlink()
            return
            
        label_file = self.labels_dir / f"{self.current_image_path.stem}.txt"
        with open(label_file, 'w') as f:
            for box in self.boxes:
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
        print(f"✅ Saved {len(self.boxes)} labels to {label_file.name}")
    
    def load_image(self, index):
        if index >= len(self.image_files) or index < 0:
            print("No more images in that direction!")
            return False
            
        self.current_index = index
        self.current_image_path = self.image_files[index]
        self.current_image = cv2.imread(str(self.current_image_path))
        
        if self.current_image is None:
            print(f"Could not load image: {self.current_image_path}")
            return False
        
        # Resize image if it's too large
        height, width = self.current_image.shape[:2]
        if height > 800 or width > 800:
            scale = min(800/width, 800/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            self.current_image = cv2.resize(self.current_image, (new_width, new_height))
        
        # Load existing labels if they exist
        label_file = self.labels_dir / f"{self.current_image_path.stem}.txt"
        self.boxes = []
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        self.boxes.append([int(parts[0])] + [float(x) for x in parts[1:]])
        
        print(f"\n📁 Loaded: {self.current_image_path.name}")
        print(f"📦 Existing boxes: {len(self.boxes)}")
        self.draw_image()
        return True
    
    def run(self):
        if not self.image_files:
            print("❌ No images found in directory!")
            print(f"Looking in: {self.image_dir}")
            return
        
        window_title = f'YOLO Annotator - Click and drag around {self.cone_type}s'
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_title, self.mouse_callback)
        
        self.load_image(0)
        
        print("\n" + "="*50)
        print(f"🟡 YOLO {self.cone_type.title()} Annotator")
        print("="*50)
        print(f"🖱️  Click and drag to create boxes around {self.cone_type}s")
        print("⏭️  Press 'n' or 'd' for next image")
        print("⏮️  Press 'p' or 'a' for previous image") 
        print("💾 Press 's' to save current annotations")
        print("🗑️  Press 'r' to remove last box")
        print("🧹 Press 'c' to clear all boxes")
        print("❌ Press 'q' or 'ESC' to quit")
        print("="*50)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q or ESC
                self.save_labels()
                break
            elif key == ord('n') or key == ord('d'):  # Next image
                self.save_labels()
                if self.load_image(self.current_index + 1):
                    pass
            elif key == ord('p') or key == ord('a'):  # Previous image
                self.save_labels()
                self.load_image(self.current_index - 1)
            elif key == ord('s'):  # Save
                self.save_labels()
            elif key == ord('r'):  # Remove last box
                if self.boxes:
                    removed = self.boxes.pop()
                    self.draw_image()
                    print(f"🗑️ Removed box #{len(self.boxes)+1}")
            elif key == ord('c'):  # Clear all boxes
                self.boxes = []
                self.draw_image()
                print("🧹 Cleared all boxes")
        
        cv2.destroyAllWindows()
        print(f"\n✅ Annotation complete! Labels saved in: {self.labels_dir}")

# Usage - Update these paths!
if __name__ == "__main__":
    # UPDATE THESE PATHS TO MATCH YOUR SETUP
    image_directory = r"C:\Users\Joshv\Desktop\Presentation\Dataset\orangeconesdataset\images\train"
    labels_directory = r"C:\Users\Joshv\Desktop\Presentation\Dataset\orangeconesdataset\labels\train"
    
    # Check if directories exist
    if not Path(image_directory).exists():
        print(f"❌ Image directory not found: {image_directory}")
        print("Please create the directory and add your images!")
        input("Press Enter to exit...")
        exit()
    
    # Create labels directory if it doesn't exist
    Path(labels_directory).mkdir(parents=True, exist_ok=True)
    
    # Initialize annotator for yellow cones
    annotator = SimpleYOLOAnnotator(image_directory, labels_directory, cone_type="Orangecone")
    annotator.run()