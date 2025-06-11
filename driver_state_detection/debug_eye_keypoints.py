#!/usr/bin/env python3
"""
Debug script to help identify the correct eye keypoint indices for EAR calculation
"""

import time
import sys
import os
import cv2
import numpy as np

# Add the current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from keypoint_model import KeypointModel
from Eye_Dector_Module import EyeDetector

def debug_eye_keypoints():
    """
    Debug the eye keypoint detection and EAR calculation
    """
    
    print("=" * 60)
    print("EYE KEYPOINT DEBUG")
    print("=" * 60)
    
    # Load keypoint model
    try:
        keypoint_model = KeypointModel('../models/outputs/model.pth')
        print("✓ Keypoint model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load keypoint model: {e}")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Cannot open camera")
        return
    
    print("✓ Camera opened successfully")
    print("\\nInstructions:")
    print("- Look at the camera with your eyes open")
    print("- Press 's' to save current frame and analyze keypoints")
    print("- Press 'c' to close your eyes and check EAR")
    print("- Press 'q' to quit")
    
    eye_detector = EyeDetector()
    frame_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 2)
        
        # Simple face detection using OpenCV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:
            # Take the biggest face
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces[0]
            
            # Add padding
            padding = 7
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.shape[0] > 1 and face_img.shape[1] > 1:
                try:
                    # Predict keypoints
                    keypoints = keypoint_model.predict(face_img)
                    
                    # Map keypoints back to original image coordinates
                    keypoints[:, 0] += x1
                    keypoints[:, 1] += y1
                    
                    # Calculate EAR
                    ear = eye_detector.get_EAR(frame=gray, landmarks=keypoints)
                    
                    # Draw all keypoints
                    for i, (kx, ky) in enumerate(keypoints):
                        kx, ky = int(kx), int(ky)
                        cv2.circle(frame, (kx, ky), 2, (0, 255, 255), -1)
                        cv2.putText(frame, str(i), (kx, ky), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 0), 1)
                    
                    # Highlight eye regions (assuming dlib indices 36-47)
                    for i in range(36, 48):  # Both eyes
                        if i < len(keypoints):
                            kx, ky = int(keypoints[i, 0]), int(keypoints[i, 1])
                            color = (0, 255, 0) if i < 42 else (255, 0, 0)  # Green for left, blue for right
                            cv2.circle(frame, (kx, ky), 4, color, 2)
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                    
                    # Display EAR
                    cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30), 
                               cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    
                    # Display threshold info
                    threshold = 0.15
                    status = "CLOSED" if ear < threshold else "OPEN"
                    color = (0, 0, 255) if ear < threshold else (0, 255, 0)
                    cv2.putText(frame, f"Eyes: {status} (thresh: {threshold})", (10, 70), 
                               cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                               
                except Exception as e:
                    print(f"Error processing keypoints: {e}")
        
        cv2.imshow("Eye Keypoint Debug - Press 's' to analyze, 'q' to quit", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            frame_counter += 1
            print(f"\\n--- FRAME {frame_counter} ANALYSIS ---")
            if len(faces) > 0 and 'keypoints' in locals():
                print(f"Total keypoints detected: {len(keypoints)}")
                print(f"EAR calculated: {ear:.3f}")
                print(f"Threshold: {threshold}")
                print(f"Eye state: {status}")
                
                # Print eye keypoint coordinates
                print("\\nEye keypoints (dlib indices 36-47):")
                for i in range(36, 48):
                    if i < len(keypoints):
                        eye_name = "Left" if i < 42 else "Right"
                        local_idx = (i - 36) % 6
                        print(f"  {eye_name} eye point {local_idx}: keypoint {i} = ({keypoints[i, 0]:.1f}, {keypoints[i, 1]:.1f})")
                
                # Save debug image
                debug_filename = f"debug_keypoints_frame_{frame_counter}.jpg"
                cv2.imwrite(debug_filename, frame)
                print(f"\\nDebug image saved as: {debug_filename}")
                
        elif key == ord('c'):
            print("\\n*** CLOSE YOUR EYES NOW AND LOOK AT THE EAR VALUE ***")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    debug_eye_keypoints()
