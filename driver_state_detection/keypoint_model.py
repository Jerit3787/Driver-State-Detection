import torch
import cv2
import numpy as np
import sys
sys.path.append('../20240311_Training_a_Robust_Facial_Keypoint_Detection_Model')
from model import FaceKeypointModel

class KeypointModel:
    def __init__(self, model_path):
        self.model = FaceKeypointModel()
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, face_img):
        # Preprocess exactly as in DebuggerCafe inference scripts
        # Resize to 224x224 (not 96x96)
        img = cv2.resize(face_img, (224, 224))
        
        # Convert BGR to RGB (important!)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img / 255.0
        
        # Transpose to channels-first format (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Reshape to (68, 2) keypoints
        keypoints = output.view(-1, 2).numpy()
        
        # Scale keypoints from 224x224 back to original face image size
        keypoints[:, 0] = keypoints[:, 0] / 224 * face_img.shape[1]  # x coordinates
        keypoints[:, 1] = keypoints[:, 1] / 224 * face_img.shape[0]  # y coordinates
        
        print("Keypoints shape:", keypoints.shape)

        # --- KEYPOINT REMAPPING PLACEHOLDER ---
        # If your model's output order is different from dlib's, remap here.
        # Example: model_to_dlib[i] = dlib_index for your model's i-th keypoint
        # Fill this list with the correct mapping if needed.
        # model_to_dlib = [ ... 68 indices ... ]
        # remapped_keypoints = np.zeros_like(keypoints)
        # for i, dlib_idx in enumerate(model_to_dlib):
        #     remapped_keypoints[dlib_idx] = keypoints[i]
        # keypoints = remapped_keypoints
        # --- END REMAPPING ---

        return keypoints
