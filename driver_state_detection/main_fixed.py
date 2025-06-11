import time
import argparse

import cv2
import numpy as np

from Utils import get_face_area
from Eye_Dector_Module import EyeDetector as EyeDet
from Pose_Estimation_Module import HeadPoseEstimator as HeadPoseEst
from Attention_Scorer_Module import AttentionScorer as AttScorer
from keypoint_model import KeypointModel

# camera matrix obtained from the camera calibration script, using a 9x6 chessboard
camera_matrix = None

# distortion coefficients obtained from the camera calibration script, using a 9x6 chessboard
dist_coeffs = None


def main():

    parser = argparse.ArgumentParser(description='Driver State Detection')

    # selection the camera number, default is 0 (webcam)
    parser.add_argument('-c', '--camera', type=int,
                        default=0, metavar='', help='Camera number, default is 0 (webcam)')

    # selection of fps limit for computing time between frames
    parser.add_argument('-F', '--fps_limit', type=int, default=11, metavar='',
                        help='FPS limit, default is 11 (WARNING: if this surpasses the fps max rate reachable by your device, it will cause problems for the scores computation)')
    # TODO: add option for choose if use camera matrix and dist coeffs

    # visualisation parameters
    parser.add_argument('--show_fps', type=bool, default=True,
                        metavar='', help='Show the actual FPS of the capture stream, default is true')
    parser.add_argument('--show_proc_time', type=bool, default=True,
                        metavar='', help='Show the processing time for a single frame, default is true')
    parser.add_argument('--show_eye_proc', type=bool, default=False,
                        metavar='', help='Show the eyes processing, deafult is false')
    parser.add_argument('--show_axis', type=bool, default=True,
                        metavar='', help='Show the head pose axis, default is true')
    parser.add_argument('--verbose', type=bool, default=False,
                        metavar='', help='Prints additional info, default is false')
    parser.add_argument('--use_mtcnn', type=bool, default=True,
                        metavar='', help='Use MTCNN face detector instead of OpenCV Haar cascade for better accuracy, default is true')

    # Attention Scorer parameters (EAR, Gaze Score, Pose)
    parser.add_argument('--smooth_factor', type=float, default=0.5,
                        metavar='', help='Sets the smooth factor for the head pose estimation keypoint smoothing, default is 0.5')
    parser.add_argument('--ear_tresh', type=float, default=0.15,
                        metavar='', help='Sets the EAR threshold for the Attention Scorer, default is 0.15')
    parser.add_argument('--ear_time_tresh', type=float, default=2,                        
                        metavar='', help='Sets the EAR time (seconds) threshold for the Attention Scorer, default is 2 seconds')
    parser.add_argument('--gaze_tresh', type=float, default=0.2,
                        metavar='', help='Sets the Gaze Score threshold for the Attention Scorer, default is 0.38 (calibrated for custom keypoint model)')
    parser.add_argument('--gaze_time_tresh', type=float, default=2, metavar='',
                        help='Sets the Gaze Score time (seconds) threshold for the Attention Scorer, default is 2 seconds')
    parser.add_argument('--perclos_tresh', type=float, default=0.2, metavar='',
                        help='Sets the PERCLOS threshold for the Attention Scorer, default is 0.2 (20%% of time period)')
    parser.add_argument('--pitch_tresh', type=float, default=30,
                        metavar='', help='Sets the PITCH threshold (degrees) for the Attention Scorer, default is 30 degrees')
    parser.add_argument('--yaw_tresh', type=float, default=20,
                        metavar='', help='Sets the YAW threshold (degrees) for the Attention Scorer, default is 20 degrees')
    parser.add_argument('--roll_tresh', type=float, default=30,
                        metavar='', help='Sets the ROLL threshold (degrees) for the Attention Scorer, default is 30 degrees')
    parser.add_argument('--pose_time_tresh', type=float, default=2.5,
                        metavar='', help='Sets the Pose time threshold (seconds) for the Attention Scorer, default is 2.5 seconds')

    # parse the arguments and store them in the args variable dictionary
    args = parser.parse_args()

    if args.verbose:
        print(f"Arguments and Parameters used:\\n{args}\\n")

    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)  # set OpenCV optimization to True
        except:
            print(
                "OpenCV optimization could not be set to True, the script may be slower than expected")

    ctime = 0  # current time (used to compute FPS)
    ptime = 0  # past time (used to compute FPS)
    prev_time = 0  # previous time variable, used to set the FPS limit
    # FPS upper limit value, needed for estimating the time for each frame and increasing performances
    fps_lim = args.fps_limit
    time_lim = 1. / fps_lim  # time window for each frame taken by the webcam
    
    # previous landmarks for head pose estimation (initially set to None) (used for smoothing)
    prev_landmarks = None

    # Face detection setup
    if args.use_mtcnn:
        from facenet_pytorch import MTCNN
        # Use MTCNN for more accurate face detection (as used in DebuggerCafe model training)
        Detector = MTCNN(keep_all=True, device='cpu')
        use_mtcnn = True
        if args.verbose:
            print("Using MTCNN face detector for improved accuracy")
    else:
        # Use OpenCV's Haar cascade face detector
        Detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        use_mtcnn = False
        if args.verbose:
            print("Using OpenCV Haar cascade face detector")

    # Load your keypoint model
    keypoint_model = KeypointModel('../models/outputs/model.pth')

    Eye_det = EyeDet(show_processing=args.show_eye_proc)

    Head_pose = HeadPoseEst(show_axis=args.show_axis)

    # instantiation of the attention scorer object, with the various thresholds
    # NOTE: set verbose to True for additional printed information about the scores
    Scorer = AttScorer(fps_lim, ear_tresh=args.ear_tresh, ear_time_tresh=args.ear_time_tresh, gaze_tresh=args.gaze_tresh,
                       gaze_time_tresh=args.gaze_time_tresh, perclos_tresh=args.perclos_tresh, pitch_tresh=args.pitch_tresh, yaw_tresh=args.yaw_tresh,
                       roll_tresh=args.roll_tresh, pose_time_tresh=args.pose_time_tresh, verbose=args.verbose)

    # capture the input from the default system camera (camera number 0)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():  # if the camera can't be opened exit the program
        print("Cannot open camera")
        exit()

    while True:  # infinite loop for webcam video capture

        delta_time = time.perf_counter() - prev_time  # delta time for FPS capping
        ret, frame = cap.read()  # read a frame from the webcam

        if not ret:  # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break

         # if the frame comes from webcam, flip it so it looks like a mirror.
        if args.camera == 0:
            frame = cv2.flip(frame, 2)

        if delta_time >= time_lim:  # if the time passed is bigger or equal than the frame time, process the frame
            prev_time = time.perf_counter()

            # compute the actual frame rate per second (FPS) of the webcam video capture stream, and show it
            ctime = time.perf_counter()
            fps = 1.0 / float(ctime - ptime)
            ptime = ctime

            # start the tick counter for computing the processing time for each frame
            e1 = cv2.getTickCount()
            # transform the BGR frame in grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # apply a bilateral filter to lower noise but keep frame details
            gray = cv2.bilateralFilter(gray, 5, 10, 10)

            # Detect faces using either MTCNN or OpenCV Haar cascade
            if use_mtcnn:
                # MTCNN detection (returns bounding boxes and confidence scores)
                bounding_boxes, conf = Detector.detect(frame, landmarks=False)
                faces = []
                if bounding_boxes is not None:
                    for box in bounding_boxes:
                        x1, y1, x2, y2 = box.astype(int)
                        # Convert to (x, y, w, h) format like OpenCV
                        faces.append([x1, y1, x2-x1, y2-y1])
            else:
                # OpenCV Haar cascade detection
                faces = Detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 0:  # process the frame only if at least a face is found

                # Take the biggest face
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]
                
                # Add some padding like in DebuggerCafe inference (crop_image function)
                padding = 7
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                face_img = frame[y1:y2, x1:x2]
                
                # Skip if face is too small
                if face_img.shape[0] > 1 and face_img.shape[1] > 1:
                    keypoints = keypoint_model.predict(face_img)
                    # Map keypoints back to original image coordinates
                    keypoints[:, 0] += x1
                    keypoints[:, 1] += y1
                
                    # Show eye keypoints (adapted for numpy keypoints)
                    for n in range(keypoints.shape[0]):
                        cx, cy = int(keypoints[n, 0]), int(keypoints[n, 1])
                        cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)
                        cv2.putText(frame, str(n), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 0), 1)

                    # Compute EAR (adapt EyeDetector to accept numpy keypoints)
                    ear = Eye_det.get_EAR(frame=gray, landmarks=keypoints)
                    
                    # Get current time for PERCLOS calculation
                    current_time = time.time()
                    tired, perclos_score = Scorer.get_PERCLOS(ear, current_time)
                    gaze = Eye_det.get_Gaze_Score(frame=gray, landmarks=keypoints)
                    frame_det, yaw, pitch, roll = Head_pose.get_pose(
                        frame=frame, landmarks=keypoints, prev_landmarks=prev_landmarks, smoothing_factor=args.smooth_factor)
                    prev_landmarks = keypoints

                if frame_det is not None:
                    frame = frame_det

                if ear is not None:
                    cv2.putText(frame, "EAR:" + str(round(ear, 3)), (10, 50),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
                if gaze is not None:
                    cv2.putText(frame, "Gaze Score:" + str(round(gaze, 3)), (10, 80),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "PERCLOS:" + str(round(perclos_score, 3)), (10, 110),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
                if tired:
                    cv2.putText(frame, "TIRED!", (10, 280),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                asleep, looking_away, distracted = Scorer.eval_scores(ear_score=ear,
                                                                      gaze_score=gaze,
                                                                      head_roll=roll,
                                                                      head_pitch=pitch,
                                                                      head_yaw=yaw,
                                                                      )
                if asleep:
                    cv2.putText(frame, "ASLEEP!", (10, 300),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                if looking_away:
                    cv2.putText(frame, "LOOKING AWAY!", (10, 320),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                if distracted:
                    cv2.putText(frame, "DISTRACTED!", (10, 340),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # stop the tick counter for computing the processing time for each frame
            e2 = cv2.getTickCount()
            # processign time in milliseconds
            proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000
            # print fps and processing time per frame on screen
            if args.show_fps:
                cv2.putText(frame, "FPS:" + str(round(fps, 0)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2,
                            (255, 0, 255), 1)
            if args.show_proc_time:
                cv2.putText(frame, "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + 'ms', (10, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                            (255, 0, 255), 1)

            # show the frame on screen
            cv2.imshow("Press 'q' to terminate", frame)

        # if the key "q" is pressed on the keyboard, the program is terminated
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()
