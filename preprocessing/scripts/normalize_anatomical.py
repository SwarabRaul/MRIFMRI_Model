import nibabel as nib
import numpy as np
import os
import glob

# Define paths
input_dir = './data/OpenNeuro/ds000115/subjects/'  # Folder with all subject folders
output_dir = './preprocessing/outputs/normalized_anat/'  # Output directory for normalized scans

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Find all subjects' T1-weighted MRI files
anat_files = glob.glob(os.path.join(input_dir, 'sub-*/anat/*_T1w.nii.gz'))

# Loop over each anatomical file and normalize
for anat_path in anat_files:
    # Load MRI image
    anat_img = nib.load(anat_path)
    anat_data = anat_img.get_fdata()

    # Normalize intensity values to [0, 1]
    anat_data_normalized = (anat_data - np.min(anat_data)) / (np.max(anat_data) - np.min(anat_data))

    # Define output path, preserving the intended folder structure without duplicating `anat`
    subject_id = os.path.basename(os.path.dirname(os.path.dirname(anat_path)))  # Extract subject ID (e.g., "sub-66")
    subject_output_dir = os.path.join(output_dir, subject_id, 'anat')
    os.makedirs(subject_output_dir, exist_ok=True)
    output_path = os.path.join(subject_output_dir, os.path.basename(anat_path))

    # Save the normalized data as a new .nii.gz file
    anat_img_normalized = nib.Nifti1Image(anat_data_normalized, anat_img.affine)
    nib.save(anat_img_normalized, output_path)

    print(f"Normalized MRI for {subject_id} saved to {output_path}")
