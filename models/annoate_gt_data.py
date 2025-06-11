"""
A simple script to draw the keypoints on the original dataset 
to check the correctness.
"""

import torch
import cv2
import os
import numpy as np
import pandas as pd

from facenet_pytorch import MTCNN
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

root_path = 'input/new_data'
input_path = 'test'
out_dir = os.path.join('outputs', input_path)
os.makedirs(out_dir, exist_ok=True)

valid_samples = pd.read_csv(os.path.join(root_path, input_path+'.csv'))

# create the MTCNN model, `keep_all=True` returns all the detected faces 
mtcnn = MTCNN(keep_all=True, device=device)

for i in tqdm(range(len(valid_samples)), total=len(valid_samples)):
    try:
        image_name = valid_samples.iloc[i][0]

        image = cv2.imread(f"{root_path}/{input_path}/{image_name}")

        orig_image = image.copy()

        bounding_boxes, conf = mtcnn.detect(image, landmarks=False)
        
        keypoints = valid_samples.iloc[i][1:]
        keypoints = np.array(keypoints, dtype='float32')
        # reshape the keypoints
        keypoints = keypoints.reshape(-1, 2)

        for j in range(len(keypoints)):
            cv2.circle(
                image, 
                center=(int(keypoints[j][0]), int(keypoints[j][1])),
                radius=2,
                color=(0, 255, 0),
                thickness=-1,
                lineType=cv2.LINE_AA
            )
            
        cv2.imwrite(os.path.join(out_dir, image_name), image)
    except:
        continue