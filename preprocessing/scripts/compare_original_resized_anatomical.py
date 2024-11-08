import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths to original normalized and resized MRI files
subject_id = 'sub-66'  # Change this to the ID of any subject you wish to inspect
original_path = f'./preprocessing/outputs/normalized_anat/{subject_id}/anat/{subject_id}_T1w.nii.gz'
resized_path = f'./preprocessing/outputs/resized_anat/{subject_id}/anat/{subject_id}_T1w.nii.gz'

# Load the original and resized images
original_img = nib.load(original_path)
resized_img = nib.load(resized_path)
original_data = original_img.get_fdata()
resized_data = resized_img.get_fdata()

# Function to display slices side-by-side
def show_comparison_slices(original_slices, resized_slices, title="MRI Comparison: Original vs Resized"):
    fig, axes = plt.subplots(2, len(original_slices), figsize=(12, 8))
    fig.suptitle(title, fontsize=16)
    
    for i, (orig_slice, res_slice) in enumerate(zip(original_slices, resized_slices)):
        axes[0, i].imshow(orig_slice.T, cmap="gray", origin="lower")
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        axes[1, i].imshow(res_slice.T, cmap="gray", origin="lower")
        axes[1, i].set_title("Resized")
        axes[1, i].axis('off')
    
    plt.show()

# Select slices from each axis
original_slices = [
    original_data[original_data.shape[0] // 2, :, :],  # Sagittal
    original_data[:, original_data.shape[1] // 2, :],  # Coronal
    original_data[:, :, original_data.shape[2] // 2]   # Axial
]

resized_slices = [
    resized_data[resized_data.shape[0] // 2, :, :],  # Sagittal
    resized_data[:, resized_data.shape[1] // 2, :],  # Coronal
    resized_data[:, :, resized_data.shape[2] // 2]   # Axial
]

# Display the comparison
show_comparison_slices(original_slices, resized_slices, title=f"MRI Comparison: Original vs Resized for {subject_id}")
