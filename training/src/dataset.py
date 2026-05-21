# PyTorch Dataset class for loading preprocessed NIfTI volumes + labels.
# Inputs: manifest CSV (patient_id, npy_path, type_label, grade_label)
# Outputs: (volume_tensor, label) pairs for DataLoader
