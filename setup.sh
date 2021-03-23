#!/bin/bash

# create virtual environment and activate it 
python3 -m venv venv
source venv/bin/activate

# install required libraries
pip install -r requirements.txt
pip install -e git+https://github.com/MattShannon/bandmat.git#egg=bandmat
