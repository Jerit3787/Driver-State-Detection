"""
Quick Gaze Threshold Tester
Run this to test different ga                    # Highlight the current threshold (0.38)
                    current_color = (0, 0, 255) if gaze_score > 0.38 else (0, 255, 0)
                    cv2.putText(frame, f"CURRENT (0.38): {'LOOKING AWAY!' if gaze_score > 0.38 else 'Normal'}", 
                               (10, y_pos + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, current_color, 2)hresholds in real-time
"""
import cv2
import numpy as np
from keypoint_model import KeypointModel
from Eye_Dector_Module import EyeDetector

def test_gaze_thresholds():
    # Load the model
    model = KeypointModel('../models/outputs/model.pth')
    eye_detector = EyeDetector()
    
    # Use webcam
    cap = cv2.VideoCapture(1)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("=== Gaze Threshold Tester ===")
    print("Current threshold: 0.38")
    print("Press 'q' to quit")
    print("Look straight, then look away to test...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 10, 10)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces[0]
            
            padding = 7
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.shape[0] > 1 and face_img.shape[1] > 1:
                keypoints = model.predict(face_img)
                keypoints[:, 0] += x1
                keypoints[:, 1] += y1
                
                gaze_score = eye_detector.get_Gaze_Score(frame=gray, landmarks=keypoints)
                
                if gaze_score is not None:                    # Test different thresholds - updated range for improved detection
                    thresholds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.38, 0.4, 0.45]
                    
                    cv2.putText(frame, f"Gaze Score: {gaze_score:.3f}", (10, 30),
                               cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    
                    y_pos = 60
                    for thresh in thresholds:
                        color = (0, 0, 255) if gaze_score > thresh else (0, 255, 0)
                        text = f"Thresh {thresh}: {'LOOKING AWAY!' if gaze_score > thresh else 'Normal'}"
                        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_PLAIN, 1.2, color, 1)
                        y_pos += 25
                      # Highlight the current threshold (0.38)
                    current_color = (0, 0, 255) if gaze_score > 0.38 else (0, 255, 0)
                    cv2.putText(frame, f"CURRENT (0.38): {'LOOKING AWAY!' if gaze_score > 0.38 else 'Normal'}", 
                               (10, y_pos + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, current_color, 2)
                else:
                    cv2.putText(frame, "Gaze Score: None", (10, 30),
                               cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        cv2.imshow('Gaze Threshold Test - Press q to quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_gaze_thresholds()
