import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import pandas as pd
import numpy as np
import os
from anatomical_3d_cnn import Anatomical3DCNN

# Define dataset
class MRIDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0], 'anat', f"{self.labels_df.iloc[idx, 0]}_T1w.nii.gz")
        image = nib.load(img_name).get_fdata()
        image = np.expand_dims(image, axis=0)  # Add channel dimension
        label = 1 if self.labels_df.iloc[idx, 1] == 'schizophrenia' else 0
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Set parameters
batch_size = 2
learning_rate = 0.001
num_epochs = 10

# Load data
train_data = MRIDataset(csv_file='project_directory/preprocessing/outputs/dataset_split_binary/train.csv',
                        root_dir='project_directory/preprocessing/outputs/resized_anat/')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_data = MRIDataset(csv_file='project_directory/preprocessing/outputs/dataset_split_binary/val.csv',
                      root_dir='project_directory/preprocessing/outputs/resized_anat/')
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = Anatomical3DCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(train_loader.dataset)

# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item() * images.size(0)
            preds = (outputs.squeeze() >= 0.5).float()
            correct += (preds == labels).sum().item()
    accuracy = correct / len(val_loader.dataset)
    return val_loss / len(val_loader.dataset), accuracy

# Run training and validation
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss, val_accuracy = validate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
