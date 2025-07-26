#!/usr/bin/env python3
"""
Face Mask Detection System Setup Script

This script automates the setup process for the Face Mask Detection System.
It handles dependency installation, model downloading, and directory creation.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def print_banner():
    """Print setup banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                Face Mask Detection System                 ║
    ║                      Setup Script                        ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    directories = [
        "static/uploads",
        "static/results", 
        "models",
        "templates"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    return True

def download_models():
    """Download pre-trained models"""
    print("🤖 Downloading pre-trained models...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Face detection model files
    files_to_download = [
        {
            "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "path": models_dir / "deploy.prototxt",
            "name": "Face Detection Config"
        },
        {
            "url": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            "path": models_dir / "res10_300x300_ssd_iter_140000.caffemodel",
            "name": "Face Detection Model"
        }
    ]
    
    for file_info in files_to_download:
        if file_info["path"].exists():
            print(f"   ✅ {file_info['name']} already exists")
            continue
            
        try:
            print(f"   📥 Downloading {file_info['name']}...")
            urllib.request.urlretrieve(file_info["url"], file_info["path"])
            print(f"   ✅ Downloaded: {file_info['name']}")
        except Exception as e:
            print(f"   ❌ Error downloading {file_info['name']}: {e}")
            return False
    
    return True

def create_env_file():
    """Create environment configuration file"""
    print("⚙️  Creating environment configuration...")
    
    env_content = """# Face Mask Detection System Configuration
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=True

# Upload settings
MAX_CONTENT_LENGTH=16777216  # 16MB
UPLOAD_FOLDER=static/uploads
RESULTS_FOLDER=static/results

# Model settings
FACE_MODEL_PROTOTXT=models/deploy.prototxt
FACE_MODEL_WEIGHTS=models/res10_300x300_ssd_iter_140000.caffemodel
MASK_MODEL_PATH=models/mask_detector.model

# Server settings
HOST=0.0.0.0
PORT=5000
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("   ✅ Created .env configuration file")

def create_run_script():
    """Create convenient run scripts"""
    print("🚀 Creating run scripts...")
    
    # Windows batch file
    batch_content = """@echo off
echo Starting Face Mask Detection System...
python app.py
pause
"""
    
    with open("run.bat", "w") as f:
        f.write(batch_content)
    
    # Unix shell script
    shell_content = """#!/bin/bash
echo "Starting Face Mask Detection System..."
python app.py
"""
    
    with open("run.sh", "w") as f:
        f.write(shell_content)
    
    # Make shell script executable
    os.chmod("run.sh", 0o755)
    
    print("   ✅ Created run.bat (Windows)")
    print("   ✅ Created run.sh (Unix/Linux/macOS)")

def test_installation():
    """Test if the installation is working"""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import cv2
        import tensorflow as tf
        import flask
        import numpy as np
        print("   ✅ All required packages imported successfully")
        
        # Test model files
        prototxt_path = Path("models/deploy.prototxt")
        weights_path = Path("models/res10_300x300_ssd_iter_140000.caffemodel")
        
        if prototxt_path.exists() and weights_path.exists():
            print("   ✅ Model files are present")
        else:
            print("   ⚠️  Warning: Some model files are missing")
            
        # Test OpenCV
        net = cv2.dnn.readNet(str(prototxt_path), str(weights_path))
        print("   ✅ OpenCV DNN model loaded successfully")
        
        return True
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def print_success_message():
    """Print success message and instructions"""
    message = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                    🎉 Setup Complete! 🎉                  ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Your Face Mask Detection System is ready to use!
    
    📋 Next Steps:
    
    1. Run the application:
       • Windows: Double-click run.bat
       • Unix/Linux/macOS: ./run.sh
       • Manual: python app.py
    
    2. Open your browser and go to:
       http://localhost:5000
    
    3. Start detecting masks:
       • Upload images on the home page
       • Try real-time detection with webcam
       • View detection history
    
    📚 Documentation:
       • README.md - Complete documentation
       • Troubleshooting section for common issues
    
    🔧 Optional Enhancements:
       • Train your own model: python train_model.py --action train
       • Customize settings in .env file
    
    💡 Tips:
       • Ensure good lighting for better detection
       • Allow camera permissions for webcam detection
       • Check the history page for detailed logs
    
    Happy detecting! 🎭
    """
    print(message)

def main():
    """Main setup function"""
    print_banner()
    
    try:
        # Step 1: Check Python version
        check_python_version()
        
        # Step 2: Create directories
        create_directories()
        
        # Step 3: Install dependencies
        if not install_dependencies():
            print("❌ Setup failed during dependency installation")
            sys.exit(1)
        
        # Step 4: Download models
        if not download_models():
            print("❌ Setup failed during model download")
            sys.exit(1)
        
        # Step 5: Create configuration
        create_env_file()
        
        # Step 6: Create run scripts
        create_run_script()
        
        # Step 7: Test installation
        if not test_installation():
            print("⚠️  Setup completed with warnings. Please check the test results.")
        
        # Step 8: Success message
        print_success_message()
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()