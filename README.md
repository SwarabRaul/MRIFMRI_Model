# Steps to run this model:

1. python preprocessing/scripts/load_and_visualize_anatomical.py 
2. python preprocessing/scripts/normalize_all_anatomical.py
3. python preprocessing/scripts/resize_anatomical.py
3.1. python preprocessing/scripts/compare_original_resized_anatomical.py



1. python preprocessing/scripts/load_functional_mri.py
2. python preprocessing/scripts/align_functional_scans.py
3. python preprocessing/scripts/normalize_functional.py



1. python preprocessing/scripts/extract_labels.py
2. python preprocessing/scripts/split_dataset.py
3. python preprocessing/scripts/split_dataset_binary.py



1. python eda/visualize_anatomical_scans.py
2. python eda/time_series_analysis.py
3. python eda/overview_time_series_analysis.py
4. python eda/task_based_functional_responses.py



1. python train/train_anatomical_cnn_tf.py

