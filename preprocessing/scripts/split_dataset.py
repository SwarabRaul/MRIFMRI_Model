import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define paths
labels_path = './preprocessing/outputs/labels.csv'
output_dir = './preprocessing/outputs/dataset_split/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load labels
labels_df = pd.read_csv(labels_path)

# Stratified split for training, validation, and test sets
train_df, temp_df = train_test_split(labels_df, test_size=0.3, stratify=labels_df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# Save each split
train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

print(f"Training set saved to {output_dir}/train.csv with {len(train_df)} samples")
print(f"Validation set saved to {output_dir}/val.csv with {len(val_df)} samples")
print(f"Test set saved to {output_dir}/test.csv with {len(test_df)} samples")
