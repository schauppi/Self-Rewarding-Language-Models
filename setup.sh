#!/bin/bash

# Create a Python 3.11 virtual environment
python3.11 -m venv self_rewarding_language_models

# Activate the virtual environment
source self_rewarding_language_models/bin/activate

# Install requirements from requirements.txt
pip install -r requirements.txt

# Download the dataset
git clone https://huggingface.co/datasets/schauppi/srlm

# Create a directory for the data
mkdir data

# Create results directories
mkdir results

# Move the dataset to the data directory
mv srlm/srlm_ift.jsonl data

# Remove the dataset directory
rm -rf srlm