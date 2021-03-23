#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "Successfully created virtual environment."

echo "Installing required libraries..."
pip install -r requirements.txt
pip install -e git+https://github.com/MattShannon/bandmat.git#egg=bandmat
echo "Successfully completed installation."
