import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define paths for the fMRI scan and events file (using sub-66 and 0-back task as an example)
subject_id = 'sub-66'
task = '0back'
func_mri_path = f'./data/OpenNeuro/ds000115/subjects/{subject_id}/func/{subject_id}_task-letter{task}task_bold.nii.gz'
events_path = f'./data/OpenNeuro/ds000115/subjects/{subject_id}/func/{subject_id}_task-letter{task}task_events.tsv'

# Load functional MRI scan
func_img = nib.load(func_mri_path)
func_data = func_img.get_fdata()

# Load events file
events_df = pd.read_csv(events_path, sep='\t')

# Print basic information about fMRI data
print("Functional MRI Data Shape:", func_data.shape)  # Expect (x, y, z, time)
print("Events Data Sample:\n", events_df.head())     # Display first few rows of the events file

# Visualize one slice over time
def plot_fmri_time_series(func_data, slice_index, time_points=10):
    fig, axes = plt.subplots(1, time_points, figsize=(15, 5))
    for i in range(time_points):
        axes[i].imshow(func_data[:, :, slice_index, i], cmap="gray")
        axes[i].axis('off')
        axes[i].set_title(f"Time {i}")
    plt.show()

# Display slices over time for a sample slice
slice_index = func_data.shape[2] // 2  # Middle axial slice
plot_fmri_time_series(func_data, slice_index)
