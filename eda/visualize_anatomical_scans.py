import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import os
import random

# Define paths
split_path = './preprocessing/outputs/dataset_split_binary/'
anat_dir = './preprocessing/outputs/resized_anat/'

# Load train split to get sample subjects
train_df = pd.read_csv(os.path.join(split_path, 'train.csv'))

# Separate subjects by label
schizophrenia_subjects = train_df[train_df['label'] == 'schizophrenia']['subject_id'].tolist()
control_subjects = train_df[train_df['label'] == 'control']['subject_id'].tolist()

# Select a few random subjects from each class for visualization
sample_schizophrenia = random.sample(schizophrenia_subjects, 3)
sample_control = random.sample(control_subjects, 3)

# Combine the samples and determine labels
samples = [(subject_id, "schizophrenia") for subject_id in sample_schizophrenia] + \
          [(subject_id, "control") for subject_id in sample_control]

# Set up the figure for displaying all subjects simultaneously
fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
fig.suptitle("Anatomical MRI Slices for Schizophrenia and Control Subjects", fontsize=16)

# Iterate over each subject and plot their slices in the grid
for i, (subject_id, label) in enumerate(samples):
    # Load anatomical MRI scan
    anat_path = os.path.join(anat_dir, subject_id, 'anat', f'{subject_id}_T1w.nii.gz')
    anat_img = nib.load(anat_path)
    anat_data = anat_img.get_fdata()

    # Select slices from each axis
    slice_0 = anat_data[anat_data.shape[0] // 2, :, :]  # Sagittal slice
    slice_1 = anat_data[:, anat_data.shape[1] // 2, :]  # Coronal slice
    slice_2 = anat_data[:, :, anat_data.shape[2] // 2]  # Axial slice

    # Plot each slice in the row for this subject
    axes[i, 0].imshow(slice_0.T, cmap="gray", origin="lower")
    axes[i, 0].set_title(f"{label.capitalize()} - {subject_id}\nSagittal")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(slice_1.T, cmap="gray", origin="lower")
    axes[i, 1].set_title("Coronal")
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(slice_2.T, cmap="gray", origin="lower")
    axes[i, 2].set_title("Axial")
    axes[i, 2].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to fit title
plt.show()
