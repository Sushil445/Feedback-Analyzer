#!/bin/bash
# Installation script for Customer Feedback Analysis System

echo "=================================="
echo "Setting up Customer Feedback Analysis System"
echo "=================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version - change if GPU available)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Download spaCy model
echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo "=================================="
echo "✓ Installation completed successfully!"
echo "=================================="
echo ""
echo "To activate the environment:"
echo "  On Linux/Mac: source venv/bin/activate"
echo "  On Windows: venv\Scripts\activate"
echo ""
echo "To run the Streamlit app:"
echo "  streamlit run streamlit_app.py"
echo ""
