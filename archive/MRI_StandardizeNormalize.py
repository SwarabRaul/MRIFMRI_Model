# import os
# import nibabel as nib
# import numpy as np

# # Define paths
# root_dir = '../../data/OpenNeuro/ds005073/subjects'  # Update with your actual root path
# output_dir = '../../preprocessing/outputs/ds005073'

# # Create output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# def preprocess_mri(mri_path, output_path):
#     # Load MRI file
#     mri_img = nib.load(mri_path)
#     mri_data = mri_img.get_fdata()
    
#     # Normalize to [0, 1]
#     mri_data_normalized = (mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))
    
#     # Save the normalized data as .npy file
#     np.save(output_path, mri_data_normalized)
#     print(f"Saved preprocessed MRI to {output_path}")

# # Loop through each subject folder and preprocess MRI
# for subject in os.listdir(root_dir):
#     subject_path = os.path.join(root_dir, subject, 'anat')
#     if os.path.isdir(subject_path):
#         # Define path to the T1-weighted scan
#         mri_path = os.path.join(subject_path, f"{subject}_rec-1_T1w.nii.gz")
        
#         # Check if the T1-weighted file exists
#         if os.path.exists(mri_path):
#             # Define the output path
#             output_path = os.path.join(output_dir, f"{subject}_T1w_normalized.npy")
            
#             # Preprocess and save
#             preprocess_mri(mri_path, output_path)
#         else:
#             print(f"File not found: {mri_path}")


# # VERIFY PREPROCESSED DATA

# # # Load a sample preprocessed scan
# # sample_scan = np.load('./preprocessing/outputs/ds005073/sub-S49_T1w_normalized.npy')
# # print("Sample scan shape:", sample_scan.shape)
# # print("Sample scan min:", np.min(sample_scan), "max:", np.max(sample_scan))

import os
import nibabel as nib
import numpy as np
from skimage.transform import resize

# Define paths
root_dir = '../../data/OpenNeuro/ds005073/subjects'  # Update with your actual root path
output_dir = '../../preprocessing/outputs/ds005073'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def preprocess_mri(mri_path, output_path, target_shape=(128, 128, 128)):
    # Load MRI file
    mri_img = nib.load(mri_path)
    mri_data = mri_img.get_fdata()
    
    # Resize the MRI data
    mri_data_resized = resize(mri_data, target_shape, mode='constant', anti_aliasing=True)
    
    # Normalize to [0, 1]
    mri_data_normalized = (mri_data_resized - np.min(mri_data_resized)) / (np.max(mri_data_resized) - np.min(mri_data_resized))
    
    # Save the normalized and resized data as .npy file
    np.save(output_path, mri_data_normalized)
    print(f"Saved preprocessed MRI to {output_path}")

# Loop through each subject folder and preprocess MRI
for subject in os.listdir(root_dir):
    subject_path = os.path.join(root_dir, subject, 'anat')
    if os.path.isdir(subject_path):
        # Define path to the T1-weighted scan
        mri_path = os.path.join(subject_path, f"{subject}_rec-1_T1w.nii.gz")
        
        # Check if the T1-weighted file exists
        if os.path.exists(mri_path):
            # Define the output path
            output_path = os.path.join(output_dir, f"{subject}_T1w_normalized.npy")
            
            # Preprocess and save
            preprocess_mri(mri_path, output_path)
        else:
            print(f"File not found: {mri_path}")

# VERIFY PREPROCESSED DATA

# # Load a sample preprocessed scan
# sample_scan = np.load('./preprocessing/outputs/ds005073/sub-S49_T1w_normalized.npy')
# print("Sample scan shape:", sample_scan.shape)
# print("Sample scan min:", np.min(sample_scan), "max:", np.max(sample_scan))
