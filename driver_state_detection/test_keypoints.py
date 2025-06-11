import cv2
import numpy as np
from keypoint_model import KeypointModel

def test_keypoint_accuracy():
    """Test the corrected keypoint preprocessing"""
    # Load the model
    model = KeypointModel('../models/outputs/model.pth')
    
    # Use webcam to capture a frame for testing
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Could not capture frame")
        cap.release()
        return
      # Use MTCNN for face detection
    from facenet_pytorch import MTCNN
    detector = MTCNN(keep_all=True, device='cpu')
    
    # MTCNN detection
    bounding_boxes, conf = detector.detect(frame, landmarks=False)
    faces = []
    if bounding_boxes is not None:
        for box in bounding_boxes:
            x1, y1, x2, y2 = box.astype(int)
            faces.append([x1, y1, x2-x1, y2-y1])
    
    if len(faces) > 0:
        # Take the largest face
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = faces[0]
        face_img = frame[y:y+h, x:x+w]
        
        print(f"Face detected: {w}x{h} pixels")
        
        # Test keypoint prediction with corrected preprocessing
        keypoints = model.predict(face_img)
        
        # Map keypoints back to original image coordinates
        keypoints[:, 0] += x
        keypoints[:, 1] += y
        
        # Draw keypoints on the frame
        for i, (x_pt, y_pt) in enumerate(keypoints):
            cv2.circle(frame, (int(x_pt), int(y_pt)), 3, (0, 255, 0), -1)
            # Only show index for first few points to avoid clutter
            if i < 10:
                cv2.putText(frame, str(i), (int(x_pt), int(y_pt)), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)
        
        # Display the result
        cv2.imshow('Keypoint Test - Press any key to exit', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("Test completed! Keypoints should be more accurate with:")
        print("- Corrected input size: 224x224 (instead of 96x96)")
        print("- BGR to RGB conversion")
        print("- Proper scaling back to original face dimensions")
        
    else:
        print("No face detected in the frame")
    
    cap.release()

if __name__ == "__main__":
    test_keypoint_accuracy()
