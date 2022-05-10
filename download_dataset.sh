#!/bin/bash

# Download dataset using kaggle API
# https://github.com/Kaggle/kaggle-api
# pip install kaggle
# ~/.kaggle/kaggle.json might be needed


# Dataset will be downloaded in 'input' directory
mkdir input
cd input

# 1. Accepted, rejected dataset
kaggle datasets download wordsforthewise/lending-club
unzip *
rm *
mv */* .
rmdir *

# 2. Feature description
wget https://resources.lendingclub.com/LCDataDictionary.xlsx
