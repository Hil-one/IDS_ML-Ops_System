#!/bin/bash
# Setup script for IDS ML-Ops System

echo "Setting up IDS ML-Ops System..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! Activate the virtual environment with: source venv/bin/activate"
