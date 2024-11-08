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
sample_schizophrenia = random.sample(schizophrenia_subjects, 3)
sample_control = random.sample(control_subjects, 3)

# Define tasks to analyze
tasks = ['0back', '1back', '2back']

# Function to calculate and plot the average signal intensity for each task
def plot_task_based_responses():
    plt.figure(figsize=(12, 8))
    plt.suptitle("Average Task-Based Functional Responses", fontsize=16)
    
    for i, (subject_id, label) in enumerate([(subj, "schizophrenia") for subj in sample_schizophrenia] +
                                            [(subj, "control") for subj in sample_control]):
        plt.subplot(2, 3, i+1)
        
        # Calculate and plot the average response for each task
        for task in tasks:
            func_path = os.path.join(func_dir, subject_id, 'func', f'{subject_id}_task-letter{task}task_normalized.nii.gz')
            func_img = nib.load(func_path)
            func_data = func_img.get_fdata()

            # Calculate the mean signal intensity across all voxels at each time point
            mean_signal = func_data.mean(axis=(0, 1, 2))
            plt.plot(mean_signal, label=f"{task} Task")
        
        plt.xlabel("Time (TRs)")
        plt.ylabel("Average Signal Intensity")
        plt.title(f"{label.capitalize()} - {subject_id}")
        plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Execute the visualization for task-based functional responses
plot_task_based_responses()
