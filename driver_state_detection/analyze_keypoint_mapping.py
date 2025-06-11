#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keypoint_model import KeypointModel
from Eye_Dector_Module import EyeDetector
import argparse
import sys

def draw_all_keypoints_with_labels(image, keypoints):
    """Draw all 68 keypoints with their index labels"""
    result = image.copy()
    
    # Draw all keypoints with different colors for different regions
    for i, (x, y) in enumerate(keypoints):
        x, y = int(x), int(y)
        
        # Color coding based on facial regions (approximate dlib mapping)
        if 0 <= i <= 16:  # Jaw line
            color = (255, 0, 0)  # Blue
        elif 17 <= i <= 21:  # Right eyebrow
            color = (0, 255, 0)  # Green
        elif 22 <= i <= 26:  # Left eyebrow
            color = (0, 255, 0)  # Green
        elif 27 <= i <= 35:  # Nose
            color = (0, 0, 255)  # Red
        elif 36 <= i <= 41:  # Right eye (dlib convention)
            color = (255, 255, 0)  # Cyan
        elif 42 <= i <= 47:  # Left eye (dlib convention)
            color = (255, 0, 255)  # Magenta
        elif 48 <= i <= 67:  # Mouth
            color = (0, 255, 255)  # Yellow
        else:
            color = (128, 128, 128)  # Gray
            
        # Draw circle for keypoint
        cv2.circle(result, (x, y), 3, color, -1)
        
        # Draw index label
        cv2.putText(result, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    return result

def analyze_eye_regions(keypoints):
    """Analyze different potential eye regions"""
    print("\n" + "="*60)
    print("DETAILED KEYPOINT ANALYSIS")
    print("="*60)
    
    # Analyze standard dlib eye regions
    print("\n1. STANDARD DLIB EYE MAPPING:")
    print("   Right eye (indices 36-41):")
    for i in range(36, 42):
        if i < len(keypoints):
            x, y = keypoints[i]
            print(f"     Point {i}: ({x:.1f}, {y:.1f})")
    
    print("   Left eye (indices 42-47):")
    for i in range(42, 48):
        if i < len(keypoints):
            x, y = keypoints[i]
            print(f"     Point {i}: ({x:.1f}, {y:.1f})")
    
    # Calculate EAR with standard mapping
    eye_detector = EyeDetector()
    standard_ear = eye_detector.get_EAR(keypoints)
    print(f"   EAR with standard mapping: {standard_ear:.3f}")
    
    # Analyze other potential eye regions
    print("\n2. ALTERNATIVE MAPPINGS TO TEST:")
    
    # Test different potential eye regions
    potential_eye_regions = [
        ("0-11", list(range(0, 12))),
        ("17-35", list(range(17, 36))),
        ("48-67", list(range(48, 68))),
    ]
    
    for name, indices in potential_eye_regions:
        if len(indices) >= 12:  # Need at least 12 points for 2 eyes
            print(f"   Testing region {name}:")
            left_eye_indices = indices[:6]
            right_eye_indices = indices[6:12]
            
            try:
                left_eye_points = [keypoints[i] for i in left_eye_indices]
                right_eye_points = [keypoints[i] for i in right_eye_indices]
                
                # Calculate EAR for this mapping
                left_ear = calculate_eye_aspect_ratio(left_eye_points)
                right_ear = calculate_eye_aspect_ratio(right_eye_points)
                avg_ear = (left_ear + right_ear) / 2
                
                print(f"     Left indices {left_eye_indices}: EAR = {left_ear:.3f}")
                print(f"     Right indices {right_eye_indices}: EAR = {right_ear:.3f}")
                print(f"     Average EAR: {avg_ear:.3f}")
            except:
                print(f"     Could not calculate EAR for region {name}")

def calculate_eye_aspect_ratio(eye_points):
    """Calculate EAR for a set of 6 eye points"""
    if len(eye_points) != 6:
        return 0.0
    
    # Convert to numpy array
    eye = np.array(eye_points)
    
    # Calculate the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    
    # Calculate the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])
    
    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def find_likely_eye_keypoints(keypoints, image_shape):
    """Try to identify which keypoints are likely to be eyes based on position"""
    print("\n3. POSITION-BASED ANALYSIS:")
    height, width = image_shape[:2]
    
    # Eyes are typically in the upper half and middle third of the face
    eye_region_y_min = height * 0.2
    eye_region_y_max = height * 0.6
    eye_region_x_min = width * 0.2
    eye_region_x_max = width * 0.8
    
    eye_candidates = []
    for i, (x, y) in enumerate(keypoints):
        if (eye_region_x_min <= x <= eye_region_x_max and 
            eye_region_y_min <= y <= eye_region_y_max):
            eye_candidates.append((i, x, y))
    
    print(f"   Keypoints in eye region (y: {eye_region_y_min:.0f}-{eye_region_y_max:.0f}, x: {eye_region_x_min:.0f}-{eye_region_x_max:.0f}):")
    for i, x, y in eye_candidates:
        print(f"     Point {i}: ({x:.1f}, {y:.1f})")
    
    return eye_candidates

def main():
    parser = argparse.ArgumentParser(description='Analyze keypoint mapping for eye detection')
    parser.add_argument('--save_analysis', action='store_true', help='Save detailed analysis images')
    args = parser.parse_args()
    
    # Initialize the keypoint model
    keypoint_model = KeypointModel()
    print("Loading keypoint model...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("\nKeypoint Mapping Analysis Tool")
    print("=" * 50)
    print("Instructions:")
    print("- Position your face in front of the camera")
    print("- Press 's' to capture and analyze current frame")
    print("- Try with eyes open and closed")
    print("- Press 'q' to quit")
    print("=" * 50)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get keypoints
        try:
            keypoints = keypoint_model.get_keypoints(frame)
            if keypoints is not None and len(keypoints) > 0:
                # Draw keypoints on frame
                display_frame = draw_all_keypoints_with_labels(frame, keypoints)
                
                # Add instruction text
                cv2.putText(display_frame, "Press 's' to analyze, 'q' to quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Keypoints detected: {len(keypoints)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Keypoint Analysis', display_frame)
            else:
                cv2.putText(frame, "No keypoints detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Keypoint Analysis', frame)
        except Exception as e:
            print(f"Error processing frame: {e}")
            cv2.imshow('Keypoint Analysis', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') and keypoints is not None:
            frame_count += 1
            print(f"\n{'='*60}")
            print(f"ANALYZING FRAME {frame_count}")
            print('='*60)
            
            # Detailed analysis
            analyze_eye_regions(keypoints)
            find_likely_eye_keypoints(keypoints, frame.shape)
            
            if args.save_analysis:
                # Save analysis image
                analysis_image = draw_all_keypoints_with_labels(frame, keypoints)
                filename = f"keypoint_analysis_frame_{frame_count}.jpg"
                cv2.imwrite(filename, analysis_image)
                print(f"\nAnalysis image saved: {filename}")
                
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
