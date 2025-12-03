#!/bin/bash

# Netflix Business Analytics Dashboard - Quick Start Script

echo "================================================"
echo "Netflix Business Analytics Dashboard"
echo "The Business Behind the Stream"
echo "================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python detected: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✓ pip detected"
echo ""

# Install requirements
echo "📦 Installing required packages..."
pip3 install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo "✓ All packages installed successfully"
else
    echo "❌ Failed to install packages. Please check requirements.txt"
    exit 1
fi

echo ""
echo "🚀 Starting Netflix Analytics Dashboard..."
echo ""
echo "The dashboard will open in your default browser at:"
echo "http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Run the dashboard
streamlit run new-net.py
