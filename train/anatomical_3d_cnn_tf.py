import tensorflow as tf
from tensorflow.keras import layers, models

def create_3d_cnn():
    inputs = tf.keras.Input(shape=(128, 128, 128, 1))  # Define input shape explicitly
    
    # First 3D Convolutional layer
    x = layers.Conv3D(32, (3, 3, 3), activation='relu')(inputs)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    # Second 3D Convolutional layer
    x = layers.Conv3D(64, (3, 3, 3), activation='relu')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    # Third 3D Convolutional layer
    x = layers.Conv3D(128, (3, 3, 3), activation='relu')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    # Flatten and Fully Connected Layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary classification output
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
