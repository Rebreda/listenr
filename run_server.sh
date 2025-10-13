#!/bin/bash

# Function to cleanup and exit
cleanup() {
    echo "Shutting down server..."
    pkill -f "gunicorn.*webapp:app"
    exit 0
}

# Set up trap for cleanup on script exit
trap cleanup EXIT INT TERM

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Kill any existing gunicorn processes
pkill -f "gunicorn.*webapp:app"

# Function to setup virtual environment
setup_venv() {
    echo "Setting up virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip

    # Check for ffmpeg installation
    if ! command -v ffmpeg &> /dev/null; then
        echo "ffmpeg is required but not installed."
        if command -v dnf &> /dev/null; then
            echo "Installing ffmpeg using dnf..."
            sudo dnf install -y ffmpeg
        elif command -v apt-get &> /dev/null; then
            echo "Installing ffmpeg using apt..."
            sudo apt-get update && sudo apt-get install -y ffmpeg
        else
            echo "Error: Could not install ffmpeg. Please install it manually."
            exit 1
        fi
    fi

    # Install web server dependencies first
    echo "Installing core web dependencies..."
    pip install flask gunicorn pydub soundfile
}

# Get local IP address (first non-localhost IPv4 address)
IP_ADDR=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '^127\.' | head -n 1)

if [ -z "$IP_ADDR" ]; then
    echo "Error: Could not determine local IP address"
    exit 1
fi

echo "Starting Listenr Web Server..."

# Check if venv exists and create if it doesn't
if [ ! -d "venv" ]; then
    setup_venv
else
    source venv/bin/activate
    
    # Verify flask and gunicorn are installed in venv
    if ! python -c "import flask, gunicorn" 2>/dev/null; then
        echo "Missing required packages in virtual environment."
        echo "Reinstalling dependencies..."
        setup_venv
    fi
fi

# Update requirements if needed
pip install -r requirements.txt

echo "Server will be available at: http://$IP_ADDR:5000"
echo "Starting server..."

# Check if port 5000 is already in use
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "Warning: Port 5000 is already in use. Attempting to free it..."
    kill $(lsof -t -i:5000) 2>/dev/null || true
    sleep 2
fi

echo "Testing connectivity..."
echo "Checking localhost access..."
if curl -s localhost:5000 >/dev/null; then
    echo "✓ localhost:5000 is accessible"
else
    echo "✗ localhost:5000 is not responding"
fi

# Start the server with gunicorn binding to all interfaces
./venv/bin/gunicorn --bind 0.0.0.0:5000 \
                    --workers 2 \
                    --timeout 120 \
                    --access-logfile - \
                    --error-logfile - \
                    --log-level debug \
                    --reload \
                    webapp:app