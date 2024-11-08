import nibabel as nib
import pandas as pd
import numpy as np
import os
import glob

# Define paths
input_dir = './data/OpenNeuro/ds000115/subjects/'
output_dir = './preprocessing/outputs/aligned_func/'
os.makedirs(output_dir, exist_ok=True)

# Define tasks to process
tasks = ['0back', '1back', '2back']

# Loop over each subject folder
subject_dirs = glob.glob(os.path.join(input_dir, 'sub-*'))

for subject_dir in subject_dirs:
    subject_id = os.path.basename(subject_dir)

    # Loop over each task
    for task in tasks:
        # Define paths for the fMRI scan and events file
        func_mri_path = os.path.join(subject_dir, 'func', f'{subject_id}_task-letter{task}task_bold.nii.gz')
        events_path = os.path.join(subject_dir, 'func', f'{subject_id}_task-letter{task}task_events.tsv')

        # Check if both files exist
        if not (os.path.exists(func_mri_path) and os.path.exists(events_path)):
            print(f"Skipping {subject_id} {task} - missing data.")
            continue

        # Load functional MRI scan and events file
        func_img = nib.load(func_mri_path)
        func_data = func_img.get_fdata()
        events_df = pd.read_csv(events_path, sep='\t')

        # Aligning (or saving, if no further processing is required)
        # Here we’re simply saving, but this step is where you'd process alignment

        # Define output path and create directories as needed
        subject_output_dir = os.path.join(output_dir, subject_id, 'func')
        os.makedirs(subject_output_dir, exist_ok=True)
        output_path = os.path.join(subject_output_dir, f'{subject_id}_task-letter{task}task_aligned.nii.gz')

        # Save the aligned data as a new .nii.gz file
        aligned_img = nib.Nifti1Image(func_data, func_img.affine)
        nib.save(aligned_img, output_path)

        print(f"Aligned fMRI data for {subject_id} {task} saved to {output_path}")
