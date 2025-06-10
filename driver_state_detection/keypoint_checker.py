import cv2
import numpy as np
from keypoint_model import KeypointModel

def check_keypoint_mapping():
    """
    Check if the custom model's keypoint order matches dlib's 68-point standard
    Specifically check if eye landmarks (36-47) are in the correct positions
    """
    # Load the model
    model = KeypointModel('../20240311_Training_a_Robust_Facial_Keypoint_Detection_Model/outputs/model.pth')
    
    # Use webcam
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    # Simple face detection using OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("=== Keypoint Mapping Checker ===")
    print("Green circles = Eye keypoints (36-47 in dlib standard)")
    print("Blue circles = Other facial keypoints") 
    print("Check if green circles are positioned correctly on your eyes")
    print("Press 'q' to quit")
    print()
    
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
                
                # Draw all keypoints with different colors for eyes
                for n in range(keypoints.shape[0]):
                    cx, cy = int(keypoints[n, 0]), int(keypoints[n, 1])
                    
                    if 36 <= n <= 47:  # Eye keypoints in dlib standard
                        # Green for eye keypoints
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                        cv2.putText(frame, str(n), (cx-10, cy-10), 
                                  cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
                    else:
                        # Blue for other keypoints
                        cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)
                        if n % 5 == 0:  # Only show some numbers to avoid clutter
                            cv2.putText(frame, str(n), (cx-10, cy-10), 
                                      cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 0, 0), 1)
                
                # Add instructions
                cv2.putText(frame, "GREEN = Eye keypoints (36-47)", (10, 30),
                           cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.putText(frame, "BLUE = Other facial keypoints", (10, 60),
                           cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
                
                # Check if eye keypoints are actually on the eyes
                left_eye_center = np.mean(keypoints[36:42], axis=0)
                right_eye_center = np.mean(keypoints[42:48], axis=0)
                
                cv2.circle(frame, tuple(left_eye_center.astype(int)), 5, (0, 255, 255), 2)
                cv2.circle(frame, tuple(right_eye_center.astype(int)), 5, (0, 255, 255), 2)
                
                cv2.putText(frame, "YELLOW = Eye centers", (10, 90),
                           cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
        
        cv2.imshow('Keypoint Mapping Check - Press q to quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_keypoint_mapping()
