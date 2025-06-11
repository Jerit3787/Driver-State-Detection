import torch

# Paths.
ROOT_PATH = 'input/new_data'
OUTPUT_PATH = 'outputs'

# Training parameters.
BATCH_SIZE = 32
LR = 0.0001
EPOCHS = 100
DEVICE = torch.device('cuda')

# Show dataset keypoint plot.
SHOW_DATASET_PLOT = False
