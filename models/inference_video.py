import torch
import numpy as np
import cv2
import config
import time
import argparse
import os

from model import FaceKeypointModel
from facenet_pytorch import MTCNN

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    help='path to input video',
    default='input/inference_videos/video_1.mp4'
)
args = parser.parse_args()

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

out_dir = os.path.join('outputs', 'video_inference')
os.makedirs(out_dir, exist_ok=True)

model = FaceKeypointModel(pretrained=False, requires_grad=False).to(device)
# load the model checkpoint
checkpoint = torch.load('outputs/model.pth', device)
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# create the MTCNN model, `keep_all=True` returns all the detected faces 
mtcnn = MTCNN(keep_all=True, device=device)

# capture the webcam
cap = cv2.VideoCapture(args.input)
if (cap.isOpened() == False):
    print('Error while trying to open video. Plese check again...')
 
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(5))

# set up the save file path
file_name = args.input.split(os.path.sep)[-1]
save_path = f"{out_dir}/{file_name}"
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"{save_path}", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 
                      frame_fps, 
                      (frame_width, frame_height))

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.

while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # Increment frame count.
        frame_count += 1
        with torch.no_grad():
            start_time = time.time()
            bounding_boxes, conf = mtcnn.detect(frame, landmarks=False)

            # Detect keypoints if face is detected.
            if bounding_boxes is not None:
                for box in bounding_boxes:
                    cropped_image, x1, y1 = crop_image(box, frame)

                    image = cropped_image.copy()

                    if image.shape[0] > 1 and image.shape[1] > 1:
                        image = cv2.resize(image, (224, 224))

                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = image / 255.0
                        image = np.transpose(image, (2, 0, 1))
                        image = torch.tensor(image, dtype=torch.float)
                        image = image.unsqueeze(0).to(config.DEVICE)

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
                                frame, 
                                (int(p[0]), int(p[1])),
                                2, 
                                (0, 0, 255), 
                                -1, 
                                cv2.LINE_AA
                            )
                
            end_time = time.time()

            fps = 1 / (end_time - start_time)
            # Add `fps` to `total_fps`.
            total_fps += fps

            cv2.putText(
                frame,
                text=f"FPS: {fps:.1f}",
                org=(15, 25),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                thickness=2,
                color=(0, 255, 0),
                lineType=cv2.LINE_AA
            )

            cv2.imshow('Facial Keypoint Frame', frame)
            out.write(frame)

            # press `q` to exit
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
    
    else: 
        break

# release VideoCapture()
cap.release()
 
# close all frames and video windows
cv2.destroyAllWindows()

# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
