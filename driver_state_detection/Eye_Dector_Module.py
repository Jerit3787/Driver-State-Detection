import cv2
import numpy as np
from numpy import linalg as LA
from Utils import resize


class EyeDetector:

    def __init__(self, show_processing: bool = False):
        """
        Eye dector class that contains various method for eye aperture rate estimation and gaze score estimation

        Parameters
        ----------
        show_processing: bool
            If set to True, shows frame images during the processing in some steps (default is False)

        Methods
        ----------
        - show_eye_keypoints: shows eye keypoints in the frame/image
        - get_EAR: computes EAR average score for the two eyes of the face
        - get_Gaze_Score: computes the Gaze_Score (normalized euclidean distance between center of eye and pupil)
            of the eyes of the face
        """

        self.keypoints = None
        self.frame = None
        self.show_processing = show_processing
        self.eye_width = None

    def show_eye_keypoints(self, color_frame, landmarks):
        """
        Shows eyes keypoints found in the face, drawing red circles in their position in the frame/image

        Parameters
        ----------
        color_frame: numpy array
            Frame/image in which the eyes keypoints are found
        landmarks: list
            List of 68 dlib keypoints of the face
        """

    
        self.keypoints = landmarks

        for n in range(36, 48):
            x = self.keypoints.part(n).x
            y = self.keypoints.part(n).y
            cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
        return

    def get_EAR(self, frame, landmarks):
        """
        Computes the average eye aperture rate of the face

        Parameters
        ----------
        frame: numpy array
            Frame/image in which the eyes keypoints are found
        landmarks: list
            List of 68 dlib keypoints of the face

        Returns
        -------- 
        ear_score: float
            EAR average score between the two eyes
            The EAR or Eye Aspect Ratio is computed as the eye opennes divided by the eye lenght
            Each eye has his scores and the two scores are averaged
        """

        self.keypoints = landmarks
        self.frame = frame
        pts = self.keypoints

        i = 0  # auxiliary counter
        # numpy array for storing the keypoints positions of the left eye
        eye_pts_l = np.zeros(shape=(6, 2))
        # numpy array for storing the keypoints positions of the right eye
        eye_pts_r = np.zeros(shape=(6, 2))

        # Support both dlib and numpy array landmarks
        for n in range(36, 42):
            if hasattr(pts, 'part'):
                point_l = pts.part(n)
                point_r = pts.part(n + 6)
                eye_pts_l[i] = [point_l.x, point_l.y]
                eye_pts_r[i] = [point_r.x, point_r.y]
            else:
                eye_pts_l[i] = [pts[n, 0], pts[n, 1]]
                eye_pts_r[i] = [pts[n + 6, 0], pts[n + 6, 1]]
            i += 1

        def EAR_eye(eye_pts):
            """
            Computer the EAR score for a single eyes given it's keypoints
            :param eye_pts: numpy array of shape (6,2) containing the keypoints of an eye considering the dlib ordering
            :return: ear_eye
                EAR of the eye
            """
            ear_eye = (LA.norm(eye_pts[1] - eye_pts[5]) + LA.norm(
                eye_pts[2] - eye_pts[4])) / (2 * LA.norm(eye_pts[0] - eye_pts[3]))
            '''
            EAR is computed as the mean of two measures of eye opening (see dlib face keypoints for the eye)
            divided by the eye lenght
            '''
            return ear_eye

        ear_left = EAR_eye(eye_pts_l)  # computing the left eye EAR score
        ear_right = EAR_eye(eye_pts_r)  # computing the right eye EAR score

        # computing the average EAR score
        ear_avg = (ear_left + ear_right) / 2

        return ear_avg

    def get_Gaze_Score(self, frame, landmarks):
        """
        Computes the average Gaze Score for the eyes
        The Gaze Score is the mean of the l2 norm (euclidean distance) between the center point of the Eye ROI
        (eye bounding box) and the center of the eye-pupil

        Parameters
        ----------
        frame: numpy array
            Frame/image in which the eyes keypoints are found
        landmarks: list
            List of 68 dlib keypoints of the face

        Returns
        -------- 
        avg_gaze_score: float
            If successful, returns the float gaze score
            If unsuccessful, returns None

        """
        self.keypoints = landmarks
        self.frame = frame

        def get_ROI(left_corner_keypoint_num: int):
            """
            Get the ROI bounding box of the eye given one of its dlib keypoints found in the face

            :param left_corner_keypoint_num: most left dlib keypoint of the eye
            :return: eye_roi
                Sub-frame of the eye region of the opencv frame/image
            """

            kp_num = left_corner_keypoint_num
            # Support both dlib and numpy array landmarks
            if hasattr(self.keypoints, 'part'):
                eye_array = np.array([
                    (self.keypoints.part(kp_num).x, self.keypoints.part(kp_num).y),
                    (self.keypoints.part(kp_num+1).x, self.keypoints.part(kp_num+1).y),
                    (self.keypoints.part(kp_num+2).x, self.keypoints.part(kp_num+2).y),
                    (self.keypoints.part(kp_num+3).x, self.keypoints.part(kp_num+3).y),
                    (self.keypoints.part(kp_num+4).x, self.keypoints.part(kp_num+4).y),
                    (self.keypoints.part(kp_num+5).x, self.keypoints.part(kp_num+5).y)
                ], np.int32)
            else:
                eye_array = np.array([
                    (self.keypoints[kp_num, 0], self.keypoints[kp_num, 1]),
                    (self.keypoints[kp_num+1, 0], self.keypoints[kp_num+1, 1]),
                    (self.keypoints[kp_num+2, 0], self.keypoints[kp_num+2, 1]),
                    (self.keypoints[kp_num+3, 0], self.keypoints[kp_num+3, 1]),
                    (self.keypoints[kp_num+4, 0], self.keypoints[kp_num+4, 1]),
                    (self.keypoints[kp_num+5, 0], self.keypoints[kp_num+5, 1])
                ], np.int32)

            min_x = np.min(eye_array[:, 0])
            max_x = np.max(eye_array[:, 0])
            min_y = np.min(eye_array[:, 1])
            max_y = np.max(eye_array[:, 1])

            eye_roi = self.frame[min_y-2:max_y+2, min_x-2:max_x+2]

            return eye_roi

        def get_gaze(eye_roi):
            """
            Computes the L2 norm between the center point of the Eye ROI
            (eye bounding box) and the center of the eye pupil
            :param eye_roi: float
            :return: (gaze_score, eye_roi): tuple
                tuple
            """

            eye_center = np.array(
                [(eye_roi.shape[1] // 2), (eye_roi.shape[0] // 2)])  # eye ROI center position
            gaze_score = None
            circles = None            # Enhanced preprocessing for better pupil detection
            eye_roi = cv2.bilateralFilter(eye_roi, 5, 80, 80)
            
            # Apply histogram equalization to improve contrast
            eye_roi = cv2.equalizeHist(eye_roi)
            
            # Try multiple parameter sets for Hough Circle detection
            circles = None
            
            # Parameter set 1: More sensitive detection
            circles = cv2.HoughCircles(eye_roi, cv2.HOUGH_GRADIENT, 1, 10,
                                       param1=50, param2=8, minRadius=1, maxRadius=15)
            
            # Parameter set 2: If first attempt fails, try with different parameters
            if circles is None or len(circles) == 0:
                circles = cv2.HoughCircles(eye_roi, cv2.HOUGH_GRADIENT, 1, 5,
                                           param1=30, param2=10, minRadius=2, maxRadius=20)
            
            # Parameter set 3: Most permissive detection
            if circles is None or len(circles) == 0:
                circles = cv2.HoughCircles(eye_roi, cv2.HOUGH_GRADIENT, 2, 5,
                                           param1=20, param2=15, minRadius=1, maxRadius=25)            # a Hough Transform is used to find the iris circle and his center (the pupil) on the grayscale eye_roi image with the contours drawn in white

            if circles is not None and len(circles) > 0:
                circles = np.uint16(np.around(circles))
                
                # Select the best circle (closest to center if multiple found)
                if len(circles[0]) > 1:
                    best_circle = None
                    min_distance = float('inf')
                    for circle in circles[0]:
                        dist_to_center = np.sqrt((circle[0] - eye_center[0])**2 + (circle[1] - eye_center[1])**2)
                        if dist_to_center < min_distance:
                            min_distance = dist_to_center
                            best_circle = circle
                    circle = best_circle
                else:
                    circle = circles[0][0, :]

                cv2.circle(
                    eye_roi, (circle[0], circle[1]), circle[2], (255, 255, 255), 1)
                cv2.circle(
                    eye_roi, (circle[0], circle[1]), 1, (255, 255, 255), -1)

                # pupil position is the first circle center found with the Hough Transform
                pupil_position = np.array([int(circle[0]), int(circle[1])])

                cv2.line(eye_roi, (eye_center[0], eye_center[1]), (
                    pupil_position[0], pupil_position[1]), (255, 255, 255), 1)

                gaze_score = LA.norm(
                    pupil_position - eye_center) / max(eye_center[0], 1)  # Avoid division by zero
                # computes the L2 distance between the eye_center and the pupil position
            else:
                # Fallback: if no pupil detected, assume looking straight (low gaze score)
                if self.show_processing:
                    cv2.putText(eye_roi, "No pupil", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                gaze_score = 0.1  # Low score indicates looking straight            
                cv2.circle(eye_roi, (eye_center[0], eye_center[1]), 1, (0, 0, 0), -1)

            if gaze_score is not None:
                return gaze_score, eye_roi
            else:
                # Return a default gaze score if detection completely fails
                return 0.15, eye_roi

        left_eye_ROI = get_ROI(36)  # computes the ROI for the left eye
        right_eye_ROI = get_ROI(42)  # computes the ROI for the right eye

        # computes the gaze scores for the eyes
        gaze_eye_left, left_eye = get_gaze(left_eye_ROI)
        gaze_eye_right, right_eye = get_gaze(right_eye_ROI)

        # if show_processing is True, shows the eyes ROI, eye center, pupil center and line distance
        if self.show_processing and (left_eye is not None) and (right_eye is not None):
            left_eye = resize(left_eye, 1000)
            right_eye = resize(right_eye, 1000)
            cv2.imshow("left eye", left_eye)
            cv2.imshow("right eye", right_eye)        
            
        if gaze_eye_left is not None and gaze_eye_right is not None:
            # computes the average gaze score for the 2 eyes
            avg_gaze_score = (gaze_eye_left + gaze_eye_right) / 2
            return avg_gaze_score
        elif gaze_eye_left is not None:
            # Use only left eye if right eye detection failed
            return gaze_eye_left
        elif gaze_eye_right is not None:
            # Use only right eye if left eye detection failed
            return gaze_eye_right
        else:
            # Both eyes failed - return default low score (looking straight)
            return 0.15
