# Volume preprocessing and augmentation transforms.
# Preprocessing: z-score normalization on non-zero voxels, clip [-3,3], rescale [0,1], resize to 96^3.
# Augmentation (training only): random LR flip, random rotation +-10deg, intensity jitter.
