import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence

def build_model(input_shape=(128, 128, 128, 1)):
    model = models.Sequential()

    # 1st Convolutional Layer
    model.add(layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    
    # 2nd Convolutional Layer
    model.add(layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    
    # 3rd Convolutional Layer
    model.add(layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    
    # Flatten and Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout to prevent overfitting
    
    # Output Layer
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification: schizophrenia or control

    return model

# Build the model
input_shape = (128, 128, 128, 1)  # Modify if your MRI scans have a different shape
model = build_model(input_shape=input_shape)
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

class MRISequence(Sequence):
    def __init__(self, file_paths, labels, batch_size):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.file_paths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_files = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_data = []
        for file in batch_files:
            scan = np.load(file)  # Load the preprocessed .npy file
            scan = np.expand_dims(scan, axis=-1)  # Add channel dimension
            batch_data.append(scan)
        
        return np.array(batch_data), np.array(batch_labels)

# Prepare file paths and labels for train, val, and test
# (Example structure: adjust paths based on your folder structure)
train_files = [f'../../data_splits/train/schizophrenia/{f}' for f in os.listdir('../../data_splits/train/schizophrenia')] + \
              [f'../../data_splits/train/control/{f}' for f in os.listdir('../../data_splits/train/control')]
train_labels = [1] * len(os.listdir('../../data_splits/train/schizophrenia')) + \
               [0] * len(os.listdir('../../data_splits/train/control'))

val_files = [f'../../data_splits/val/schizophrenia/{f}' for f in os.listdir('../../data_splits/val/schizophrenia')] + \
            [f'../../data_splits/val/control/{f}' for f in os.listdir('../../data_splits/val/control')]
val_labels = [1] * len(os.listdir('../../data_splits/val/schizophrenia')) + \
             [0] * len(os.listdir('../../data_splits/val/control'))

test_files = [f'../../data_splits/test/schizophrenia/{f}' for f in os.listdir('../../data_splits/test/schizophrenia')] + \
             [f'../../data_splits/test/control/{f}' for f in os.listdir('../../data_splits/test/control')]
test_labels = [1] * len(os.listdir('../../data_splits/test/schizophrenia')) + \
              [0] * len(os.listdir('../../data_splits/test/control'))

# Set up the data generators
batch_size = 8
train_gen = MRISequence(train_files, train_labels, batch_size=batch_size)
val_gen = MRISequence(val_files, val_labels, batch_size=batch_size)
test_gen = MRISequence(test_files, test_labels, batch_size=batch_size)

epochs = 20  # Start with 20 epochs and adjust based on performance

history = model.fit(train_gen, epochs=epochs, validation_data=val_gen)

test_loss, test_accuracy = model.evaluate(test_gen)
print(f"Test accuracy: {test_accuracy}")