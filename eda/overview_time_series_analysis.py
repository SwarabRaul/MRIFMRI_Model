import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Define paths
split_path = './preprocessing/outputs/dataset_split_binary/'
func_dir = './preprocessing/outputs/normalized_func/'

# Load train split to get sample subjects
train_df = pd.read_csv(os.path.join(split_path, 'train.csv'))

# Separate subjects by label
schizophrenia_subjects = train_df[train_df['label'] == 'schizophrenia']['subject_id'].tolist()
control_subjects = train_df[train_df['label'] == 'control']['subject_id'].tolist()

# Select a few random subjects from each class for visualization
sample_schizophrenia = random.sample(schizophrenia_subjects, 2)
sample_control = random.sample(control_subjects, 2)

# Define tasks and voxel coordinates for the overview
tasks = ['0back', '1back', '2back']
voxel_coords = (32, 32, 18)  # Example voxel coordinates

# Function to plot combined time series for each subject across tasks
def plot_overview_time_series():
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Time-Series Overview Across Tasks", fontsize=16)
    
    for i, (subject_id, label) in enumerate([(subj, "schizophrenia") for subj in sample_schizophrenia] +
                                            [(subj, "control") for subj in sample_control]):
        plt.subplot(2, 2, i+1)
        
        # Plot time-series for each task with a different color and label
        for task in tasks:
            func_path = os.path.join(func_dir, subject_id, 'func', f'{subject_id}_task-letter{task}task_normalized.nii.gz')
            func_img = nib.load(func_path)
            func_data = func_img.get_fdata()
            
            # Extract the time-series data from the specified voxel
            voxel_time_series = func_data[voxel_coords[0], voxel_coords[1], voxel_coords[2], :]
            plt.plot(voxel_time_series, label=f"{task} Task")
        
        plt.xlabel("Time (TRs)")
        plt.ylabel("Signal Intensity")
        plt.title(f"{label.capitalize()} - {subject_id}")
        plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Execute the combined time-series analysis across all tasks
plot_overview_time_series()
