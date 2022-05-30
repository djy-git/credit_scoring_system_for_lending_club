#!/bin/bash

# Download dataset using kaggle API
# https://github.com/Kaggle/kaggle-api
# pip install kaggle
# ~/.kaggle/kaggle.json might be needed


# Dataset will be downloaded in 'input' directory
cd input

# 1. Accepted, rejected dataset
kaggle datasets download wordsforthewise/lending-club
unzip *.zip
mv */* .
rmdir *
rm *.zip *.csv.gz

# 2. Feature description (file exists in input directory)
#wget https://resources.lendingclub.com/LCDataDictionary.xlsx
