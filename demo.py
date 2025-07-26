#!/usr/bin/env python3
"""
Face Mask Detection Demo Script

This script demonstrates the face mask detection functionality
without the web interface. Useful for testing and development.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def load_face_detector():
    """Load the face detection model"""
    prototxt_path = "models/deploy.prototxt"
    weights_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
    
    if Path(prototxt_path).exists() and Path(weights_path).exists():
        net = cv2.dnn.readNet(prototxt_path, weights_path)
        return net
    else:
        # Fallback to Haar Cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        return face_cascade

def detect_faces_dnn(image, net, confidence_threshold=0.5):
    """Detect faces using DNN model"""
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence < confidence_threshold:
            continue
            
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # Ensure the bounding boxes are within the frame
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w, endX)
        endY = min(h, endY)
        
        if (endX - startX) > 0 and (endY - startY) > 0:
            faces.append((startX, startY, endX, endY))
    
    return faces

def detect_faces_haar(image, cascade):
    """Detect faces using Haar Cascade"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    return [(x, y, x + w, y + h) for (x, y, w, h) in faces]

def predict_mask(face_region):
    """
    Predict if the face is wearing a mask
    This is a simplified heuristic for demo purposes
    In a real system, you would use a trained CNN model
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    
    # Simple heuristic: check for darker regions in the lower face area
    height, width = gray.shape
    lower_face = gray[int(height*0.6):, :]
    
    # Calculate average brightness of lower face
    avg_brightness = np.mean(lower_face)
    
    # If lower face is significantly darker, assume mask
    # This is a very basic heuristic and not accurate for real use
    if avg_brightness < 80:
        return "Mask", 0.7 + np.random.random() * 0.25  # Random confidence 0.7-0.95
    else:
        return "No Mask", 0.6 + np.random.random() * 0.3  # Random confidence 0.6-0.9

def process_image(image_path, output_path=None, show_result=True):
    """Process a single image for mask detection"""
    print(f"Processing image: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    # Load face detector
    detector = load_face_detector()
    
    # Detect faces
    if hasattr(detector, 'setInput'):  # DNN model
        faces = detect_faces_dnn(image, detector)
    else:  # Haar Cascade
        faces = detect_faces_haar(image, detector)
    
    print(f"Found {len(faces)} face(s)")
    
    # Process each face
    results = []
    for i, (startX, startY, endX, endY) in enumerate(faces):
        # Extract face region
        face = image[startY:endY, startX:endX]
        
        # Predict mask
        label, confidence = predict_mask(face)
        
        # Draw bounding box and label
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        
        # Add label with confidence
        label_text = f"{label}: {confidence:.1%}"
        cv2.putText(image, label_text, (startX, startY - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        results.append({
            "face": i + 1,
            "label": label,
            "confidence": confidence,
            "box": (startX, startY, endX, endY)
        })
        
        print(f"  Face {i+1}: {label} ({confidence:.1%})")
    
    # Save or show result
    if output_path:
        cv2.imwrite(str(output_path), image)
        print(f"Result saved to: {output_path}")
    
    if show_result:
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.title(f"Face Mask Detection Results - {len(faces)} face(s) detected")
        plt.axis('off')
        plt.show()
    
    return results

def process_webcam():
    """Process webcam feed for real-time detection"""
    print("Starting webcam detection...")
    print("Press 'q' to quit, 's' to save current frame")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Load face detector
    detector = load_face_detector()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        
        # Process every 3rd frame for better performance
        if frame_count % 3 == 0:
            # Detect faces
            if hasattr(detector, 'setInput'):  # DNN model
                faces = detect_faces_dnn(frame, detector)
            else:  # Haar Cascade
                faces = detect_faces_haar(frame, detector)
            
            # Process each face
            for (startX, startY, endX, endY) in faces:
                # Extract face region
                face = frame[startY:endY, startX:endX]
                
                # Predict mask
                label, confidence = predict_mask(face)
                
                # Draw bounding box and label
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
                # Add label with confidence
                label_text = f"{label}: {confidence:.1%}"
                cv2.putText(frame, label_text, (startX, startY - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Face Mask Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"demo_capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Face Mask Detection Demo")
    parser.add_argument("--mode", choices=["image", "webcam"], default="image",
                       help="Detection mode")
    parser.add_argument("--input", type=str,
                       help="Input image path (for image mode)")
    parser.add_argument("--output", type=str,
                       help="Output image path (optional)")
    parser.add_argument("--no-display", action="store_true",
                       help="Don't display the result")
    
    args = parser.parse_args()
    
    if args.mode == "image":
        if not args.input:
            # Use sample image from img directory if available
            sample_images = list(Path("img").glob("*.jpg")) + list(Path("img").glob("*.png"))
            if sample_images:
                args.input = sample_images[0]
                print(f"Using sample image: {args.input}")
            else:
                print("Error: No input image specified and no sample images found")
                print("Usage: python demo.py --mode image --input path/to/image.jpg")
                return
        
        result = process_image(
            args.input, 
            args.output, 
            show_result=not args.no_display
        )
        
        if result:
            print(f"\nDetection Summary:")
            mask_count = sum(1 for r in result if r['label'] == 'Mask')
            no_mask_count = len(result) - mask_count
            print(f"  Total faces: {len(result)}")
            print(f"  With masks: {mask_count}")
            print(f"  Without masks: {no_mask_count}")
    
    elif args.mode == "webcam":
        process_webcam()

if __name__ == "__main__":
    main()