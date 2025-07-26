import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import json
from datetime import datetime
import base64

# Optional TensorFlow imports for advanced features
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available - using basic detection only")
    TF_AVAILABLE = False

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Global variables for model and face detector
face_net = None
mask_net = None

def load_models():
    """Load the face detection and mask classification models"""
    global face_net, mask_net
    
    # Load face detection model
    prototxtPath = "models/deploy.prototxt"
    weightsPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
    
    if os.path.exists(prototxtPath) and os.path.exists(weightsPath):
        face_net = cv2.dnn.readNet(prototxtPath, weightsPath)
    else:
        # Create a simple face detector using OpenCV's built-in classifier
        face_net = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load mask detection model
    if TF_AVAILABLE and os.path.exists("models/mask_detector.model"):
        mask_net = tf.keras.models.load_model("models/mask_detector.model")
    else:
        # Create and use a simple pre-trained model
        mask_net = create_simple_model()

def create_simple_model():
    """Create a simple mask detection model"""
    if not TF_AVAILABLE:
        print("TensorFlow not available - returning None for model")
        return None
        
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Initialize with random weights for demo purposes
    # In a real scenario, you would load pre-trained weights
    return model

def detect_and_predict_mask(frame, face_net, mask_net):
    """Detect faces and predict mask/no mask"""
    (h, w) = frame.shape[:2]
    results = []
    
    if isinstance(face_net, cv2.CascadeClassifier):
        # Using Haar Cascade classifier
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_net.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w_face, h_face) in faces:
            # Extract face region
            face = frame[y:y+h_face, x:x+w_face]
            
            # Simple demo prediction based on color analysis
            if TF_AVAILABLE:
                # Use TensorFlow preprocessing if available
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (224, 224))
                face_array = img_to_array(face_resized)
                face_preprocessed = preprocess_input(face_array)
                face_batch = np.expand_dims(face_preprocessed, axis=0)
                # Would use trained model here: preds = mask_net.predict(face_batch, verbose=0)[0]
            
            # Simple heuristic-based prediction for demo
            avg_color = np.mean(face)
            mask_prob = 0.7 if avg_color < 100 else 0.3  # Darker regions suggest mask
            no_mask_prob = 1 - mask_prob
            
            preds = [mask_prob, no_mask_prob]
            (mask, withoutMask) = preds
                
            label = "Mask" if mask > withoutMask else "No Mask"
            confidence = max(mask, withoutMask)
            
            results.append({
                "box": (x, y, x + w_face, y + h_face),
                "label": label,
                "confidence": confidence
            })
    
    return results

def log_detection(filename, results):
    """Log detection results to a JSON file"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "detections": results
    }
    
    log_file = "detection_log.json"
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(log_entry)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and process image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Invalid image file'})
        
        # Detect faces and masks
        results = detect_and_predict_mask(image, face_net, mask_net)
        
        # Draw bounding boxes and labels
        for result in results:
            (startX, startY, endX, endY) = result["box"]
            label = result["label"]
            confidence = result["confidence"]
            
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label_text = f"{label}: {confidence:.2f}"
            
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            cv2.putText(image, label_text, (startX, startY - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        # Save result image
        result_filename = f"result_{filename}"
        result_path = os.path.join('static/results', result_filename)
        cv2.imwrite(result_path, image)
        
        # Log detection
        log_detection(filename, results)
        
        return jsonify({
            'success': True,
            'result_image': url_for('static', filename=f'results/{result_filename}'),
            'detections': len(results),
            'results': results
        })

@app.route('/webcam')
def webcam():
    """Webcam page"""
    return render_template('webcam.html')

def generate_frames():
    """Generate frames for webcam stream"""
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Detect faces and masks
        results = detect_and_predict_mask(frame, face_net, mask_net)
        
        # Draw bounding boxes and labels
        for result in results:
            (startX, startY, endX, endY) = result["box"]
            label = result["label"]
            confidence = result["confidence"]
            
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label_text = f"{label}: {confidence:.2f}"
            
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, label_text, (startX, startY - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def history():
    """View detection history"""
    log_file = "detection_log.json"
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
        return render_template('history.html', logs=logs)
    else:
        return render_template('history.html', logs=[])

if __name__ == '__main__':
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load models
    load_models()
    
    app.run(debug=True, host='0.0.0.0', port=5000)