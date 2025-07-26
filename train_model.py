#!/usr/bin/env python3
"""
Face Mask Detection Model Training Script

This script creates and trains a mask detection model using MobileNetV2 as the base.
For demo purposes, it creates a simple model. In production, you would:
1. Download a real mask/no-mask dataset
2. Train on actual data
3. Save the trained model
"""

import os
import argparse

# Conditional imports for training functionality
try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"Training dependencies not available: {e}")
    print("Only model download functionality will be available.")
    TRAINING_AVAILABLE = False

def create_model(input_shape=(224, 224, 3)):
    """
    Create a mask detection model using MobileNetV2 as the base
    """
    # Load the MobileNetV2 network, ensuring the head FC layer sets are left off
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                           input_tensor=Input(shape=input_shape))

    # Construct the head of the model that will be placed on top of the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    # Place the head FC model on top of the base model
    model = Model(inputs=baseModel.input, outputs=headModel)

    # Loop over all layers in the base model and freeze them so they will
    # not be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    return model

def create_synthetic_data(num_samples=1000):
    """
    Create synthetic training data for demonstration
    In a real scenario, you would load actual mask/no-mask images
    """
    print("[INFO] Creating synthetic training data...")
    
    # Create random image data
    X = np.random.rand(num_samples, 224, 224, 3) * 255
    X = X.astype("float32")
    
    # Create labels (0 = no mask, 1 = mask)
    y = np.random.randint(0, 2, num_samples)
    
    # Add some patterns to make the data slightly more realistic
    for i in range(num_samples):
        if y[i] == 1:  # Mask
            # Add darker regions to simulate mask
            X[i, 100:150, 80:180] *= 0.6
        else:  # No mask
            # Keep normal brightness
            pass
    
    return X, y

def train_model(epochs=10, learning_rate=1e-4, save_path="models/mask_detector.model"):
    """
    Train the mask detection model
    """
    if not TRAINING_AVAILABLE:
        print("[ERROR] Training dependencies not available. Please install TensorFlow and other ML libraries.")
        return None
        
    print("[INFO] Loading and preprocessing data...")
    
    # For demo purposes, create synthetic data
    # In production, replace this with real dataset loading
    X, y = create_synthetic_data(num_samples=2000)
    
    # Preprocess the data
    X = preprocess_input(X)
    
    # Convert labels to categorical
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    y = to_categorical(y)
    
    # Split the data
    (trainX, testX, trainY, testY) = train_test_split(X, y,
                                                      test_size=0.20, 
                                                      stratify=y, 
                                                      random_state=42)
    
    # Data augmentation
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    
    print("[INFO] Creating model...")
    model = create_model()
    
    # Compile the model
    opt = Adam(learning_rate=learning_rate, decay=learning_rate / epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    print("[INFO] Training model...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=32),
        steps_per_epoch=len(trainX) // 32,
        validation_data=(testX, testY),
        validation_steps=len(testX) // 32,
        epochs=epochs)
    
    print("[INFO] Evaluating model...")
    predIdxs = model.predict(testX, batch_size=32)
    predIdxs = np.argmax(predIdxs, axis=1)
    
    print(classification_report(testY.argmax(axis=1), predIdxs,
                               target_names=lb.classes_))
    
    # Save the model
    print(f"[INFO] Saving mask detector model to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path, save_format="h5")
    
    # Plot training history
    plot_training_history(H, epochs)
    
    return model

def plot_training_history(H, epochs):
    """
    Plot training accuracy and loss
    """
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("models/training_history.png")
    plt.show()

def download_pretrained_model():
    """
    Download a pre-trained face detection model
    """
    import urllib.request
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Download face detection model files
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    prototxt_path = os.path.join(models_dir, "deploy.prototxt")
    model_path = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    
    if not os.path.exists(prototxt_path):
        print("[INFO] Downloading face detection prototxt...")
        urllib.request.urlretrieve(prototxt_url, prototxt_path)
    
    if not os.path.exists(model_path):
        print("[INFO] Downloading face detection model...")
        urllib.request.urlretrieve(model_url, model_path)
    
    print("[INFO] Face detection models downloaded successfully!")

def main():
    parser = argparse.ArgumentParser(description="Train or download mask detection models")
    parser.add_argument("--action", choices=["train", "download"], default="download",
                       help="Action to perform: train new model or download pretrained")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    if args.action == "train":
        print("[INFO] Training mask detection model...")
        if TRAINING_AVAILABLE:
            train_model(epochs=args.epochs, learning_rate=args.lr)
        else:
            print("[ERROR] Training not available. Install TensorFlow and ML dependencies first.")
    elif args.action == "download":
        print("[INFO] Downloading pre-trained models...")
        download_pretrained_model()

if __name__ == "__main__":
    main()