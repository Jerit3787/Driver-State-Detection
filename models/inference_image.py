import torch
import numpy as np
import cv2
import config
import os
import glob

from model import FaceKeypointModel
from facenet_pytorch import MTCNN
from tqdm import tqdm

# Crop the detected face with padding and return it.
def crop_image(box, image, pre_padding=7, post_padding=7):
    x1, y1, x2, y2 = box
    x1 = x1 - pre_padding
    y1 = y1 - pre_padding
    x2 = x2 + post_padding
    y2 = y2 + post_padding
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    return cropped_image, x1, y1

device = torch.device('cpu')
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

out_dir = os.path.join('outputs', 'image_inference')
os.makedirs(out_dir, exist_ok=True)

model = FaceKeypointModel(pretrained=False, requires_grad=False).to(config.DEVICE)
# load the model checkpoint
checkpoint = torch.load('outputs/model.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().to(device)

# create the MTCNN model, `keep_all=True` returns all the detected faces 
mtcnn = MTCNN(keep_all=True, device=device)

input_path = 'input/inference_images'

all_image_paths = glob.glob(os.path.join(input_path, '*'))

for image_path in tqdm(all_image_paths, total=len(all_image_paths)):
    image_name = image_path.split(os.path.sep)[-1]

    orig_image = cv2.imread(image_path)

    bounding_boxes, conf = mtcnn.detect(orig_image, landmarks=False)

    # Detect keypoints only if face is detected.
    if bounding_boxes is not None:
        for box in bounding_boxes:
            cropped_image, x1, y1 = crop_image(box, orig_image)

            image = cropped_image.copy()

            if image.shape[0] > 1 and image.shape[1] > 1:
                image = cv2.resize(image, (224, 224))
            
                image = image / 255.0
            
                image = np.transpose(image, (2, 0, 1))
                image = torch.tensor(image, dtype=torch.float)
                image = image.unsqueeze(0).to(config.DEVICE)
            
                with torch.no_grad():
                    outputs = model(image)
            
                outputs = outputs.cpu().detach().numpy()
            
                outputs = outputs.reshape(-1, 2)
                keypoints = outputs

                # Draw keypoints on face.
                for i, p in enumerate(keypoints):
                    p[0] = p[0] / 224 * cropped_image.shape[1]
                    p[1] = p[1] / 224 * cropped_image.shape[0]
            
                    p[0] += x1
                    p[1] += y1
                    
                    cv2.circle(
                        orig_image, 
                        (int(p[0]), int(p[1])),
                        2, 
                        (0, 0, 255), 
                        -1, 
                        cv2.LINE_AA
                    )
                
    cv2.imwrite(os.path.join(out_dir, image_name), orig_image)