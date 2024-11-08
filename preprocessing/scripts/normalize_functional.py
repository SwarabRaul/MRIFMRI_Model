import nibabel as nib
import numpy as np
import os
import glob

# Define paths
input_dir = './preprocessing/outputs/aligned_func/'  # Input directory with aligned functional scans
output_dir = './preprocessing/outputs/normalized_func/'  # Output directory for normalized functional scans

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over each aligned functional MRI file
func_files = glob.glob(os.path.join(input_dir, 'sub-*/func/*_task-letter*task_aligned.nii.gz'))

for func_path in func_files:
    # Load functional MRI image
    func_img = nib.load(func_path)
    func_data = func_img.get_fdata()

    # Normalize intensity values to [0, 1]
    func_data_normalized = (func_data - np.min(func_data)) / (np.max(func_data) - np.min(func_data))

    # Extract the subject ID and task name from the file path to ensure unique filenames
    subject_id = os.path.basename(os.path.dirname(os.path.dirname(func_path)))  # Extract subject ID
    task_name = os.path.basename(func_path).replace("_aligned.nii.gz", "")  # Extract task-specific filename

    # Define output directory and path for each subject and task
    subject_output_dir = os.path.join(output_dir, subject_id, 'func')
    os.makedirs(subject_output_dir, exist_ok=True)
    output_path = os.path.join(subject_output_dir, f'{task_name}_normalized.nii.gz')

    # Save the normalized data as a new .nii.gz file
    func_img_normalized = nib.Nifti1Image(func_data_normalized, func_img.affine)
    nib.save(func_img_normalized, output_path)

    print(f"Normalized fMRI for {subject_id} {task_name} saved to {output_path}")
