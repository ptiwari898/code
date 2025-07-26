# Face Mask Detection System

A Flask-based web application for detecting face masks in images and real-time webcam streams using a pre-trained deep learning model (MobileNetV2 or custom CNN).

## Features
- Upload an image to detect if faces are wearing masks or not
- Real-time webcam mask detection
- Bounding boxes and confidence scores for each detected face
- Detection history logging
- Responsive Bootstrap frontend

## Requirements
- Python 3.7+
- See `requirements.txt` for Python dependencies
- Pre-trained Keras model file (`mask_detector.model`) in the project root

## Setup
1. Clone the repository or download the source code.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download or train a face mask detection model and save it as `mask_detector.model` in the project root. (You can find public models or train your own using MobileNetV2.)
4. Run the Flask app:
   ```bash
   python app.py
   ```
5. Open your browser and go to `http://127.0.0.1:5000/`

## Usage
- **Image Upload:** Upload an image to detect masks on all faces in the image. Results are shown with bounding boxes and labels.
- **Webcam Detection:** Click "Start Webcam Detection" to use your webcam for real-time mask detection.
- **Detection History:** View the log of previous detections.

## Notes
- The app requires a pre-trained Keras model (`mask_detector.model`). You can use a public model or train your own.
- The app uses OpenCV's Haar Cascade for face detection and MobileNetV2 for mask classification.

## License
MIT