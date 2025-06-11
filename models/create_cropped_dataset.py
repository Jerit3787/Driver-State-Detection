"""
Create new images folder along with new CSV files for new cropped face landmark dataset.
"""

import torch
import cv2
import os
import numpy as np
import pandas as pd

from facenet_pytorch import MTCNN
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# create the MTCNN model, `keep_all=True` returns all the detected faces 
mtcnn = MTCNN(keep_all=True, device=device)

root_path = 'input/orig_dataset'
input_paths = ['training', 'test']

pre_padding = 7
post_padding = 7

def crop_image(bounding_boxes, image):
    if bounding_boxes is not None:
        x1, y1, x2, y2 = bounding_boxes[0]
        x1 = x1 - pre_padding
        y1 = y1 - pre_padding
        x2 = x2 + post_padding
        y2 = y2 + post_padding
        cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
        return cropped_image, x1, y1
    else:
        return image, 0, 0

for input_path in input_paths:
    out_dir = os.path.join('input', 'new_data', input_path)
    os.makedirs(out_dir, exist_ok=True)
    
    valid_samples = pd.read_csv(os.path.join(root_path, input_path+'_frames_keypoints.csv'))

    image_names_list = []
    keypoints_list = []

    cols = [str(i) for i in range(136)]
    cols.insert(0, '')
    new_df = pd.DataFrame(columns=cols)
    
    for i in tqdm(range(len(valid_samples)), total=len(valid_samples)):
        try:
            image_name = valid_samples.iloc[i, 0]

            image = cv2.imread(f"{root_path}/{input_path}/{image_name}")

            orig_image = image.copy()

            bounding_boxes, conf = mtcnn.detect(image, landmarks=False)

            cropped_image, x1, y1 = crop_image(bounding_boxes, orig_image)
                
            cv2.imwrite(os.path.join(out_dir, image_name), cropped_image)
            keypoints = valid_samples.iloc[i][1:]
            keypoints = np.array(keypoints)

            # reshape the keypoints
            keypoints = keypoints.reshape(-1, 2)

            keypoints =  keypoints - [x1, y1]

            keypoints = keypoints.reshape(-1)

            keypoints = list(keypoints)
            keypoints.insert(0, image_name)

            new_df.loc[i] = keypoints
        except:
            continue
    
    print(new_df)
    new_df.to_csv('input/new_data/'+input_path+'.csv', index=False)