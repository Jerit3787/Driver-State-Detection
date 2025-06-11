import cv2
import numpy as np
from keypoint_model import KeypointModel
from Eye_Dector_Module import EyeDetector

def analyze_gaze_detection():
    """
    Diagnostic tool to analyze gaze detection with custom keypoint model
    This will help us understand what gaze scores we get for different head poses
    """
    # Load the model
    model = KeypointModel('../models/outputs/model.pth')
    eye_detector = EyeDetector(show_processing=True)  # Enable processing display
    
    # Use webcam
    cap = cv2.VideoCapture(1)  # Use camera 1 as in main.py
    
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    # Simple face detection using OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("=== Gaze Detection Analyzer ===")
    print("Instructions:")
    print("1. Look straight at the camera")
    print("2. Look left, right, up, down")
    print("3. Observe the gaze scores printed")
    print("4. Press 'q' to quit")
    print("5. Press 's' to save current gaze score")
    print()
    
    gaze_scores = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 10, 10)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) > 0:
            # Take the largest face
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
                # Get keypoints
                keypoints = model.predict(face_img)
                keypoints[:, 0] += x1
                keypoints[:, 1] += y1
                
                # Calculate gaze score
                gaze_score = eye_detector.get_Gaze_Score(frame=gray, landmarks=keypoints)
                
                # Draw keypoints (just eye region)
                for n in range(36, 48):  # Eye keypoints
                    cx, cy = int(keypoints[n, 0]), int(keypoints[n, 1])
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)
                    cv2.putText(frame, str(n), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 0), 1)
                
                # Display information
                if gaze_score is not None:
                    cv2.putText(frame, f"Gaze Score: {gaze_score:.3f}", (10, 30),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    
                    # Threshold indicators
                    cv2.putText(frame, f"Threshold: 0.4", (10, 60),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                    
                    if gaze_score > 0.4:
                        cv2.putText(frame, "WOULD TRIGGER: LOOKING AWAY!", (10, 90),
                                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Normal gaze", (10, 90),
                                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                    
                    print(f"Gaze Score: {gaze_score:.3f}")
                else:
                    cv2.putText(frame, "Gaze Score: None", (10, 30),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    print("Gaze Score: None (pupil detection failed)")
        
        cv2.imshow('Gaze Analysis - Press q to quit, s to save score', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and len(gaze_scores) > 0:
            gaze_scores.append(gaze_score if gaze_score is not None else 0)
            print(f"Saved gaze score: {gaze_score}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if gaze_scores:
        print(f"\n=== Analysis Results ===")
        print(f"Collected {len(gaze_scores)} gaze scores:")
        print(f"Min: {min(gaze_scores):.3f}")
        print(f"Max: {max(gaze_scores):.3f}")
        print(f"Average: {np.mean(gaze_scores):.3f}")
        print(f"Current threshold: 0.4")
        print(f"\nRecommended threshold based on your data: {np.mean(gaze_scores) + 0.1:.3f}")

if __name__ == "__main__":
    analyze_gaze_detection()
