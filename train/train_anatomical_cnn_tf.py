import tensorflow as tf
import nibabel as nib
import pandas as pd
import numpy as np
import os
from scipy.ndimage import zoom
from anatomical_3d_cnn_tf import create_3d_cnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set to '2' to show only warnings and errors

# Load labels
train_labels_path = './preprocessing/outputs/dataset_split_binary/train.csv'
val_labels_path = './preprocessing/outputs/dataset_split_binary/val.csv'
train_df = pd.read_csv(train_labels_path)
val_df = pd.read_csv(val_labels_path)

# Define function to load and preprocess MRI data
def load_mri_image(subject_id, label, root_dir):
    img_path = os.path.join(root_dir, subject_id, 'anat', f"{subject_id}_T1w.nii.gz")
    image = nib.load(img_path).get_fdata()
    # Resize the 3D image to (128, 128, 128) using zoom
    factors = (128 / image.shape[0], 128 / image.shape[1], 128 / image.shape[2])
    image = zoom(image, factors, order=1)  # Linear interpolation for resizing
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    label = tf.convert_to_tensor(1 if label == 'schizophrenia' else 0, dtype=tf.float32)
    return image, label

# Convert data to TensorFlow Dataset
def create_dataset(data_df, root_dir, batch_size=2):
    def generator():
        for _, row in data_df.iterrows():
            yield load_mri_image(row['subject_id'], row['label'], root_dir)
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(128, 128, 128, 1), dtype=tf.float32),  # Explicitly set image shape
        tf.TensorSpec(shape=(), dtype=tf.float32)  # Label is scalar
    ))
    dataset = dataset.batch(batch_size)
    return dataset

# Paths and parameters
root_dir = './preprocessing/outputs/resized_anat/'
batch_size = 4
learning_rate = 0.001
num_epochs =  30

# Create datasets
train_dataset = create_dataset(train_df, root_dir, batch_size)
val_dataset = create_dataset(val_df, root_dir, batch_size)

# Initialize model, compile with loss and optimizer
model = create_3d_cnn()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training the model
model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset)
