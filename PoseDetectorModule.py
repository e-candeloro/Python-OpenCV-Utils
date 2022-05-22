import mediapipe as mp
import numpy as np
import cv2

import time


class poseDetector():
    def __init__(self, mode=False, modCompl=1, upBody=False, smooth=True, segm=False, smooth_seg=True, detCon=0.5, trackCon=0.5):
        self.mode = mode  # static image mode
        self.modCompl = modCompl
        self.upBody = upBody
        self.smooth = smooth
        self.segm = segm
        self.smooth_seg = smooth_seg
        self.detCon = detCon  # detection confidence threshold
        self.trackCon = trackCon  # tracking confidence threshold

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=self.modCompl,
                                     smooth_landmarks=self.smooth,
                                     enable_segmentation=self.segm,
                                     smooth_segmentation=self.smooth_seg,
                                     min_detection_confidence=self.detCon,
                                     min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, frame, draw=True):
        '''

        :param: img (opencv image in BGR)
        :param: draw (boolean, draw the keypoint if set to true, default is true)
        :returns: img (opencv image in BGR with keypoints drawn if draw is set to true)
        '''
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return frame

    def findPosePosition(self, frame, additional_info=False, draw=True):
        '''
        Given and image, returns the pose keypoints position in the format of a list of lists
        [[id_point0, x_point0, y_point0], ...]

        Keypoints list  are shown on this site: https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png

        :param: img (opencv BGR image)
        :draw: bool (draws circles over the keypoints, default is true)

        :returns: 
            lm_list (list of lists of keypoints)
            img
        '''
        self.lm_list = []
        h, w, c = frame.shape

        if self.results.pose_landmarks:
            pose = self.results.pose_landmarks
            for id_point, lm in enumerate(pose.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if additional_info:
                    cz = lm.z
                    vis = lm.visibility
                    self.lm_list.append([id_point, cx, cy, cz, vis])
                else:
                    self.lm_list.append([id_point, cx, cy])

                if draw:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        return self.lm_list

    def findAngle(self, frame, p1, p2, p3, flip_angle = False,draw=True):
        '''Find the angle between 3 points p1, p2, p3 in succession, where p2 is the point where the angle is measured.
        For the points, only the index number is required. Please refer to this image to select the appriopriate keypoints: https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png

        Example: elbow angle, given the shoulder keypoint, the elbow keypoint and the wrist keypoint

        :param: frame (opencv frame)
        :p1:first point index
        :p2:second point index
        :p3:third point index
        :flip_angle: Bool: flips the angle computation
        :draw: Bool (optional): draws additional info, default is True

        Returns:
            -angle: angle in degrees between the segment s12 and the segment s23 having p2 as vertex, where the angle is located
        '''
        # checks if keypoints values are correct
        assert p1 >= 0 and p1 <= 32, f"p1 must be >=0 and <=32"
        assert p2 >= 0 and p2 <= 32, f"p2 must be >=0 and <=32"
        assert p3 >= 0 and p3 <= 32, f"p3 must be >=0 and <=32"

        if len(self.lm_list) > 0:
            x1, y1 = self.lm_list[p1][1:3]
            x2, y2 = self.lm_list[p2][1:3]
            x3, y3 = self.lm_list[p3][1:3]
        else:
            return None
        
        if flip_angle:
            flipped = -1
        else:
            flipped = 1

        angle = np.degrees((flipped)*np.arctan2(y3 - y2, x3 - x2) - 1*(flipped)*np.arctan2(y1 - y2, x1 - x2))

        if angle < 0:
            angle += 360

        if draw:
            cv2.circle(frame, (x2, y2), 5, (255, 0, 255), -1)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 255), 1)
            cv2.putText(frame, str(round(angle, 0)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return angle

# ---------------------------------------------------------------
# MAIN SCRIPT EXAMPLE FOR REAL-TIME POSE TRACKING USING A WEBCAM
# ---------------------------------------------------------------


def main(camera_source=0, show_fps=True, verbose=False):

    assert camera_source >= 0, f"source needs to be greater or equal than 0\n"

    ctime = 0  # current time (used to compute FPS)
    ptime = 0  # past time (used to compute FPS)

    cv2.setUseOptimized(True)

    # capture the input from the default system camera (camera number 0)
    cap = cv2.VideoCapture(camera_source)
    detector = poseDetector(detCon=0.7, trackCon=0.7)

    if not cap.isOpened():  # if the camera can't be opened exit the program
        print("Cannot open camera")
        exit()

    while True:  # infinite loop for webcam video capture

        ret, frame = cap.read()  # read a frame from the webcam

        if not ret:  # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break

        frame = detector.findPose(frame=frame)
        lm_list = detector.findPosePosition(
            frame, additional_info=True, draw=True)

        if len(lm_list) > 0:
            angle = detector.findAngle(frame, 12, 14, 16,flip_angle=True,draw=True)
        # compute the actual frame rate per second (FPS) of the webcam video capture stream, and show it
        ctime = time.perf_counter()
        fps = 1.0 / float(ctime - ptime)
        ptime = ctime

        if show_fps:
            cv2.putText(frame, "FPS:" + str(round(fps, 0)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 255, 255), 1)

        # show the frame on screen
        cv2.imshow("Frame (press 'q' to exit)", frame)

        # if the key "q" is pressed on the keyboard, the program is terminated
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    # change this to zero if you don't have a usb webcam but an in-built camera
    main(camera_source=1)
