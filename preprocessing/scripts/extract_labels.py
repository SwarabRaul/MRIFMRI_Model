import pandas as pd
import os

# Define path to the participants file and output location
participants_path = './data/OpenNeuro/ds000115/participants.tsv'
output_dir = './preprocessing/outputs/'
labels_output_path = os.path.join(output_dir, 'labels.csv')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the participants.tsv file
participants_df = pd.read_csv(participants_path, sep='\t')

# Check the first few rows to confirm the structure
print("Participants Data Sample:\n", participants_df.head())

# Extract subject IDs and condition labels from the 'condit' column
if 'condit' in participants_df.columns:
    labels_df = participants_df[['participant_id', 'condit']]
    labels_df.columns = ['subject_id', 'label']  # Rename columns for clarity
else:
    print("Error: 'condit' column not found in participants.tsv")
    exit()

# Save the labels to a CSV file for easy access
labels_df.to_csv(labels_output_path, index=False)
print(f"Labels saved to {labels_output_path}")