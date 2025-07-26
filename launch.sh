#!/bin/bash

echo "🚀 Face Mask Detection System Launcher"
echo "======================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if required files exist
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found. Please run this script from the project directory."
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found. Please run this script from the project directory."
    exit 1
fi

echo "✅ Project files found"

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "📁 Creating models directory..."
    mkdir -p models
fi

# Check if models are downloaded
if [ ! -f "models/deploy.prototxt" ] || [ ! -f "models/res10_300x300_ssd_iter_140000.caffemodel" ]; then
    echo "🤖 Downloading face detection models..."
    python3 train_model.py --action download
    if [ $? -ne 0 ]; then
        echo "❌ Failed to download models"
        exit 1
    fi
fi

echo "✅ Models ready"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p static/uploads
mkdir -p static/results
mkdir -p templates

echo "✅ Directories created"

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import flask, cv2, numpy, PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies missing. They should already be installed."
    echo "   If you encounter issues, run: python3 -m pip install --break-system-packages -r requirements.txt"
fi

echo "✅ Dependencies check complete"
echo ""

# Launch options
echo "🎯 Choose launch option:"
echo "1) Start Web Application (Recommended)"
echo "2) Run Demo with Image"
echo "3) Run Demo with Webcam"
echo "4) Exit"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🌐 Starting Face Mask Detection Web Application..."
        echo "📍 Server will be available at: http://localhost:5000"
        echo "🛑 Press Ctrl+C to stop the server"
        echo ""
        python3 app.py
        ;;
    2)
        echo ""
        read -p "📁 Enter image path (or press Enter for sample image): " image_path
        if [ -z "$image_path" ]; then
            if [ -f "img/pexels-ben-mack-5326953.jpg" ]; then
                image_path="img/pexels-ben-mack-5326953.jpg"
                echo "🖼️  Using sample image: $image_path"
            else
                echo "❌ No sample image found. Please provide an image path."
                exit 1
            fi
        fi
        echo "🔍 Running mask detection on image..."
        python3 demo.py --mode image --input "$image_path" --no-display
        ;;
    3)
        echo ""
        echo "📹 Starting webcam detection..."
        echo "🛑 Press 'q' to quit, 's' to save frame"
        python3 demo.py --mode webcam
        ;;
    4)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac