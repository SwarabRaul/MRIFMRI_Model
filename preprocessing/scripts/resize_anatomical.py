import nibabel as nib
import numpy as np
import os
import glob
from scipy.ndimage import zoom

# Define paths
input_dir = './preprocessing/outputs/normalized_anat/'  # Input directory with normalized scans
output_dir = './preprocessing/outputs/resized_anat/'  # Output directory for resized scans

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Target shape for resizing
target_shape = (128, 128, 128)

# Loop over each normalized anatomical file
anat_files = glob.glob(os.path.join(input_dir, 'sub-*/anat/*_T1w.nii.gz'))

for anat_path in anat_files:
    # Load the normalized MRI image
    anat_img = nib.load(anat_path)
    anat_data = anat_img.get_fdata()

    # Calculate resize factors for each dimension
    factors = [target_dim / orig_dim for target_dim, orig_dim in zip(target_shape, anat_data.shape)]
    
    # Resize the data
    anat_data_resized = zoom(anat_data, factors, order=1)  # Linear interpolation for resizing

    # Define output path, preserving subject-specific folder structure
    subject_id = os.path.basename(os.path.dirname(os.path.dirname(anat_path)))  # Extract subject ID
    subject_output_dir = os.path.join(output_dir, subject_id, 'anat')
    os.makedirs(subject_output_dir, exist_ok=True)
    output_path = os.path.join(subject_output_dir, os.path.basename(anat_path))

    # Save the resized data as a new .nii.gz file
    anat_img_resized = nib.Nifti1Image(anat_data_resized, anat_img.affine)
    nib.save(anat_img_resized, output_path)

    print(f"Resized MRI for {subject_id} saved to {output_path}")
