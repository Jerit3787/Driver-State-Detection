import torch
import cv2
import pandas as pd
import numpy as np
import config
import utils
import os

from torch.utils.data import Dataset, DataLoader

class FaceKeypointDataset(Dataset):
    def __init__(self, samples, path):
        self.data = samples
        self.path = path
        self.resize = 224

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = cv2.imread(f"{self.path}/{self.data.iloc[index, 0]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, channel = image.shape
        # Resize the image into `resize` defined above.
        image = cv2.resize(image, (self.resize, self.resize))

        image = image / 255.0
        # Transpose for getting the channel size to index 0.
        image = np.transpose(image, (2, 0, 1))
        # Get the keypoints.
        keypoints = self.data.iloc[index, 1:]
        keypoints = np.array(keypoints, dtype='float32')
        # Reshape the keypoints.
        keypoints = keypoints.reshape(-1, 2)
        # Rescale keypoints according to image resize.
        keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
        }
    
training_samples = pd.read_csv(os.path.join(config.ROOT_PATH, 'training.csv'))
valid_samples = pd.read_csv(os.path.join(config.ROOT_PATH, 'test.csv'))

# Initialize the dataset - `FaceKeypointDataset()`.
train_data = FaceKeypointDataset(training_samples, 
                                 f"{config.ROOT_PATH}/training")
valid_data = FaceKeypointDataset(valid_samples, 
                                 f"{config.ROOT_PATH}/test")

# Prepare data loaders.
train_loader = DataLoader(train_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=True)
valid_loader = DataLoader(valid_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=False)

print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")

# Whether to show dataset keypoint plots.
if config.SHOW_DATASET_PLOT:
    utils.dataset_keypoints_plot(valid_data)