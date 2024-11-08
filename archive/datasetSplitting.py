import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# Paths to preprocessed data and where to save splits
data_dir = '../../preprocessing/outputs/ds005073'
train_dir = '../../data_splits/train'
val_dir = '../../data_splits/val'
test_dir = '../../data_splits/test'

# Ensure directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Collect all MRI file paths and labels
data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
labels = ['schizophrenia' if 'sub-S' in f else 'control' for f in data_files]  # Adjust according to your label pattern

# Split into train, validation, and test sets
train_files, test_files, train_labels, test_labels = train_test_split(data_files, labels, test_size=0.3, stratify=labels, random_state=42)
val_files, test_files, val_labels, test_labels = train_test_split(test_files, test_labels, test_size=0.5, stratify=test_labels, random_state=42)

# Function to copy files to respective folders
def copy_files(files, labels, dest_dir):
    for f, label in zip(files, labels):
        label_dir = os.path.join(dest_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy(f, label_dir)

# Copy files to train, val, and test folders
copy_files(train_files, train_labels, train_dir)
copy_files(val_files, val_labels, val_dir)
copy_files(test_files, test_labels, test_dir)

print("Data split completed!")
