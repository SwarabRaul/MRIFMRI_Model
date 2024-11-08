import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Define path to the anatomical MRI file
anat_mri_path = './data/OpenNeuro/ds000115/subjects/sub-66/anat/sub-66_T1w.nii.gz'  # Replace with your actual path

# Load MRI image
anat_img = nib.load(anat_mri_path)
anat_data = anat_img.get_fdata()

# Print basic information
print("Data shape:", anat_data.shape)
print("Voxel dimensions (affine):\n", anat_img.affine)

# Function to display MRI slices
def show_slices(slices, title="MRI Slices"):
    fig, axes = plt.subplots(1, len(slices), figsize=(12, 4))
    fig.suptitle(title, fontsize=16)
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        axes[i].axis('off')
    plt.show()

# Select and show slices along each axis
slice_0 = anat_data[anat_data.shape[0] // 2, :, :]  # Sagittal slice
slice_1 = anat_data[:, anat_data.shape[1] // 2, :]  # Coronal slice
slice_2 = anat_data[:, :, anat_data.shape[2] // 2]  # Axial slice
show_slices([slice_0, slice_1, slice_2], title="Anatomical MRI Slices for Subject sub-66")