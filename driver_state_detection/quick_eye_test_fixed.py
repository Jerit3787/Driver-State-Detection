#!/usr/bin/env python3

import cv2
import numpy as np
from keypoint_model import KeypointModel
from Eye_Dector_Module import EyeDetector

def test_keypoint_positions():
    """Test which keypoints actually correspond to eyes"""
    
    # Initialize the keypoint model
    keypoint_model = KeypointModel('../20240311_Training_a_Robust_Facial_Keypoint_Detection_Model/outputs/model.pth')
    eye_detector = EyeDetector()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Quick Eye Keypoint Test")
    print("======================")
    print("Position your face and press SPACE to capture a frame")
    print("Press ESC to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show the frame
        cv2.putText(frame, "Press SPACE to test keypoints, ESC to exit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Eye Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space key
            try:
                # First detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                
                if len(faces) > 0:
                    # Take the biggest face
                    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                    x, y, w, h = faces[0]
                    
                    # Add padding like in main.py
                    padding = 7
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    
                    face_img = frame[y1:y2, x1:x2]
                    
                    if face_img.shape[0] > 1 and face_img.shape[1] > 1:
                        # Get keypoints using the correct method
                        keypoints = keypoint_model.predict(face_img)
                        # Map keypoints back to original image coordinates
                        keypoints[:, 0] += x1
                        keypoints[:, 1] += y1
                        
                        if keypoints is not None and len(keypoints) >= 68:
                            print(f"\nFrame captured! Analyzing {len(keypoints)} keypoints...")
                            
                            # Test current EAR calculation
                            current_ear = eye_detector.get_EAR(frame=gray, landmarks=keypoints)
                            print(f"Current EAR (using indices 36-47): {current_ear:.3f}")
                            
                            # Show face center for reference
                            height, width = frame.shape[:2]
                            center_x, center_y = width // 2, height // 2
                            print(f"Frame center: ({center_x}, {center_y})")
                            
                            # Test different potential eye regions
                            print("\nTesting different eye keypoint regions:")
                            print("-" * 50)
                            
                            # Standard dlib mapping (what we're currently using)
                            print("1. Standard dlib (36-47):")
                            left_eye_std = keypoints[36:42]
                            right_eye_std = keypoints[42:48]
                            print(f"   Left eye (36-41): {[f'({x:.0f},{y:.0f})' for x,y in left_eye_std]}")
                            print(f"   Right eye (42-47): {[f'({x:.0f},{y:.0f})' for x,y in right_eye_std]}")
                            print(f"   EAR: {current_ear:.3f}")
                            
                            # Test some other potential mappings
                            test_mappings = [
                                ("Lower indices", list(range(17, 29))),  # Eyebrow region potentially
                                ("Middle indices", list(range(27, 39))),  # Nose/eye transition
                                ("Alternative", list(range(20, 32))),     # Different potential eye region
                            ]
                            
                            for name, indices in test_mappings:
                                if len(indices) >= 12:
                                    try:
                                        left_indices = indices[:6]
                                        right_indices = indices[6:12]
                                        
                                        left_points = [keypoints[i] for i in left_indices]
                                        right_points = [keypoints[i] for i in right_indices]
                                        
                                        # Calculate custom EAR
                                        left_ear = calculate_ear(left_points)
                                        right_ear = calculate_ear(right_points)
                                        avg_ear = (left_ear + right_ear) / 2
                                        
                                        print(f"\n2. {name} ({indices[0]}-{indices[-1]}):")
                                        print(f"   Left indices {left_indices}: {[f'({keypoints[i][0]:.0f},{keypoints[i][1]:.0f})' for i in left_indices]}")
                                        print(f"   Right indices {right_indices}: {[f'({keypoints[i][0]:.0f},{keypoints[i][1]:.0f})' for i in right_indices]}")
                                        print(f"   EAR: {avg_ear:.3f}")
                                        
                                    except Exception as e:
                                        print(f"\n2. {name}: Error calculating EAR - {e}")
                            
                            # Show spatial distribution
                            print(f"\nSpatial analysis:")
                            print(f"All keypoints range:")
                            xs = [x for x, y in keypoints]
                            ys = [y for x, y in keypoints]
                            print(f"  X: {min(xs):.0f} to {max(xs):.0f}")
                            print(f"  Y: {min(ys):.0f} to {max(ys):.0f}")
                            
                            # Look for keypoints in eye region
                            eye_y_min = min(ys) + (max(ys) - min(ys)) * 0.2
                            eye_y_max = min(ys) + (max(ys) - min(ys)) * 0.5
                            eye_x_min = min(xs) + (max(xs) - min(xs)) * 0.1
                            eye_x_max = min(xs) + (max(xs) - min(xs)) * 0.9
                            
                            eye_region_points = []
                            for i, (x, y) in enumerate(keypoints):
                                if eye_x_min <= x <= eye_x_max and eye_y_min <= y <= eye_y_max:
                                    eye_region_points.append((i, x, y))
                            
                            print(f"\nKeypoints in potential eye region (y: {eye_y_min:.0f}-{eye_y_max:.0f}):")
                            for i, x, y in eye_region_points:
                                print(f"  Point {i}: ({x:.0f}, {y:.0f})")
                        else:
                            print("Insufficient keypoints detected")
                    else:
                        print("Face image too small")
                else:
                    print("No face detected")
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                
        elif key == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()

def calculate_ear(eye_points):
    """Calculate EAR for 6 eye points"""
    if len(eye_points) != 6:
        return 0.0
    
    eye = np.array(eye_points)
    
    # Vertical distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    
    # Horizontal distance
    C = np.linalg.norm(eye[0] - eye[3])
    
    if C == 0:
        return 0.0
    
    ear = (A + B) / (2.0 * C)
    return ear

if __name__ == "__main__":
    test_keypoint_positions()
