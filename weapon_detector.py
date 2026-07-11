#!/usr/bin/env python3
"""
Real-Time Weapon Detection System
Detects firearms and knives using YOLOv8
Works with webcam or CCTV feeds
"""

import cv2
import torch
from ultralytics import YOLO
from datetime import datetime
import json
import os
from pathlib import Path
import threading
import queue
from playsound import playsound
import time

class WeaponDetector:
    def __init__(self, model_name='yolov8m.pt', confidence=0.5, alert_sound=True):
        """
        Initialize weapon detector
        
        Args:
            model_name: YOLOv8 model size (nano, small, medium, large, xlarge)
            confidence: Detection confidence threshold (0-1)
            alert_sound: Enable audio alerts
        """
        self.model_name = model_name
        self.confidence = confidence
        self.alert_sound = alert_sound
        
        # Load model
        print(f"Loading YOLOv8 model: {model_name}...")
        self.model = YOLO(model_name)
        print("✓ Model loaded successfully")
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Detection classes to track (firearms and knives)
        self.weapon_classes = {
            'gun', 'rifle', 'pistol', 'revolver', 'shotgun', 'handgun',
            'knife', 'dagger', 'blade', 'sword', 'firearm', 'weapon',
            'bow', 'crossbow', 'explosive'
        }
        
        # Logging
        self.log_dir = Path('detection_logs')
        self.log_dir.mkdir(exist_ok=True)
        self.detection_log = []
        
        # Alert queue
        self.alert_queue = queue.Queue()
        
    def create_alert_sound(self):
        """Create a simple alert beep"""
        import numpy as np
        import scipy.io.wavfile as wavfile
        
        sample_rate = 44100
        duration = 0.5
        frequency = 1000  # Hz
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        wave = np.sin(2 * np.pi * frequency * t) * 0.3
        
        alert_file = 'alert.wav'
        wavfile.write(alert_file, sample_rate, (wave * 32767).astype(np.int16))
        return alert_file
    
    def play_alert(self):
        """Play alert sound in background thread"""
        if self.alert_sound:
            try:
                alert_file = self.create_alert_sound()
                playsound(alert_file)
                os.remove(alert_file)
            except:
                # Fallback: just print
                print("\n🚨 WEAPON DETECTED! 🚨\n")
    
    def log_detection(self, detections, frame_count, fps):
        """Log detection to file and memory"""
        timestamp = datetime.now()
        
        for det in detections:
            class_name = det['class']
            confidence = det['confidence']
            
            log_entry = {
                'timestamp': timestamp.isoformat(),
                'class': class_name,
                'confidence': float(confidence),
                'frame': frame_count
            }
            
            self.detection_log.append(log_entry)
            
            # Save to file
            log_file = self.log_dir / f"detections_{timestamp.strftime('%Y-%m-%d')}.json"
            with open(log_file, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')
    
    def detect_weapons_in_frame(self, frame):
        """Detect weapons in a single frame"""
        # Run inference
        results = self.model(frame, conf=self.confidence, verbose=False)
        
        weapons_detected = []
        
        # Parse results
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                
                # Check if it's a weapon
                if any(weapon in class_name.lower() for weapon in self.weapon_classes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    weapons_detected.append({
                        'class': class_name,
                        'confidence': confidence,
                        'box': (x1, y1, x2, y2)
                    })
        
        return weapons_detected
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['box']
            class_name = det['class']
            confidence = det['confidence']
            
            # Red box for weapons
            color = (0, 0, 255)  # BGR: Red
            thickness = 3
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def run_webcam(self, camera_source=0, save_video=False):
        """
        Run real-time detection on webcam feed
        
        Args:
            camera_source: 0 for default webcam, or IP camera URL
            save_video: Save detected frames to video file
        """
        print(f"\n🎥 Starting weapon detection on camera: {camera_source}")
        print("Press 'q' to quit, 's' to save frame\n")
        
        # Open camera
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            print(f"❌ Failed to open camera: {camera_source}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Video writer
        video_writer = None
        if save_video:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'weapon_detection_{timestamp}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            print(f"💾 Saving video to: {output_file}")
        
        frame_count = 0
        weapon_found_in_sequence = False
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                frame_count += 1
                
                # Resize for faster processing
                frame_resized = cv2.resize(frame, (640, 480))
                
                # Detect weapons
                detections = self.detect_weapons_in_frame(frame_resized)
                
                # Draw detections
                frame_with_boxes = self.draw_detections(frame_resized, detections)
                
                # Log detections
                if detections:
                    self.log_detection(detections, frame_count, fps)
                    weapon_found_in_sequence = True
                    
                    # Alert
                    for det in detections:
                        alert_msg = f"🚨 {det['class'].upper()} DETECTED ({det['confidence']:.2%}) 🚨"
                        print(alert_msg)
                    
                    # Play sound (in background thread)
                    alert_thread = threading.Thread(target=self.play_alert, daemon=True)
                    alert_thread.start()
                
                # Display info
                info_text = f"Frame: {frame_count} | Weapons: {len(detections)} | FPS: {fps:.1f}"
                cv2.putText(frame_with_boxes, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Weapon Detection', frame_with_boxes)
                
                # Save to video
                if video_writer:
                    video_writer.write(frame_with_boxes)
                
                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n✓ Exiting...")
                    break
                elif key == ord('s'):
                    filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame_with_boxes)
                    print(f"✓ Frame saved: {filename}")
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            print(f"\n✓ Detection finished")
            print(f"  Total frames processed: {frame_count}")
            print(f"  Total detections: {len(self.detection_log)}")
            if self.detection_log:
                print(f"  Logs saved to: {self.log_dir}")
    
    def run_image_detection(self, image_path):
        """Detect weapons in a static image"""
        print(f"\n🖼️ Detecting weapons in: {image_path}")
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Failed to load image: {image_path}")
            return
        
        # Detect
        detections = self.detect_weapons_in_frame(frame)
        
        # Draw
        frame_with_boxes = self.draw_detections(frame, detections)
        
        # Log
        if detections:
            self.log_detection(detections, 0, 0)
        
        # Display
        cv2.imshow('Detection Results', frame_with_boxes)
        
        # Save result
        output_path = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(output_path, frame_with_boxes)
        print(f"✓ Result saved: {output_path}")
        
        print(f"✓ Detections: {len(detections)}")
        for det in detections:
            print(f"  - {det['class']}: {det['confidence']:.2%}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def show_logs(self, days=1):
        """Display recent detection logs"""
        target_date = datetime.now().strftime('%Y-%m-%d')
        log_file = self.log_dir / f"detections_{target_date}.json"
        
        if not log_file.exists():
            print(f"No logs found for {target_date}")
            return
        
        print(f"\n📋 Detection Logs ({target_date}):\n")
        detections_by_class = {}
        
        with open(log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                class_name = entry['class']
                if class_name not in detections_by_class:
                    detections_by_class[class_name] = 0
                detections_by_class[class_name] += 1
        
        for class_name, count in sorted(detections_by_class.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count} detections")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Weapon Detection System')
    parser.add_argument('--mode', choices=['webcam', 'image', 'logs'], default='webcam',
                       help='Detection mode')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera source (0=default, or IP camera URL)')
    parser.add_argument('--image', type=str,
                       help='Image path for detection')
    parser.add_argument('--model', choices=['nano', 'small', 'medium', 'large', 'xlarge'], 
                       default='small',
                       help='YOLOv8 model size')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--save-video', action='store_true',
                       help='Save detected video')
    parser.add_argument('--no-sound', action='store_true',
                       help='Disable alert sounds')
    
    args = parser.parse_args()
    
    # Model mapping
    model_map = {
        'nano': 'yolov8n.pt',
        'small': 'yolov8s.pt',
        'medium': 'yolov8m.pt',
        'large': 'yolov8l.pt',
        'xlarge': 'yolov8x.pt'
    }
    
    # Initialize detector
    detector = WeaponDetector(
        model_name=model_map[args.model],
        confidence=args.confidence,
        alert_sound=not args.no_sound
    )
    
    # Run
    if args.mode == 'webcam':
        detector.run_webcam(camera_source=args.camera, save_video=args.save_video)
    elif args.mode == 'image':
        if not args.image:
            print("Error: --image path required for image mode")
            return
        detector.run_image_detection(args.image)
    elif args.mode == 'logs':
        detector.show_logs()


if __name__ == '__main__':
    main()
