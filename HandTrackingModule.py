import mediapipe as mp
import cv2
import numpy as np
import time

from OpencvUtils import draw_pose_info, rotationMatrixToEulerAngles


class HandDetector():
    def __init__(self, mode=False, maxHands=1, modCompl=1, detCon=0.5, trackCon=0.5):
        self.mode = mode  # static image mode,
        self.maxHands = maxHands  # max number of hands to track
        self.modCompl = modCompl  # complexity of the model (can be 0 or 1)
        self.detCon = detCon  # detection confidence threshold
        self.trackCon = trackCon  # tracking confidence threshold

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        model_complexity=self.modCompl,
                                        min_detection_confidence=self.detCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        '''
        Detects the hands and draws keypoints of the hands given and input image.
        :param: img (opencv image in BGR)
        :param: draw (boolean, draw the keypoint if set to true, default is true)
        :returns: img (opencv image in BGR with keypoints drawn if draw is set to true)
        '''
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLMs in self.results.multi_hand_landmarks:
                for id_point, lm in enumerate(handLMs.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
            if draw:
                self.mpDraw.draw_landmarks(img, handLMs,
                                           self.mpHands.HAND_CONNECTIONS)
        return img

    def findHandPosition(self, img, hand_num=0, draw=True):
        '''
        Given and image, returns the hand keypoints position in the format of a list of lists
        [[id_point0, x_point0, y_point0], ..., [id_point19, x_point19, y_point19]]
        The number of hand keypoints are 20 in total.
        Keypoints list and relative position are shown in the example notebook and on this site: https://google.github.io/mediapipe/solutions/hands.html

        :param: img (opencv BGR image)
        :param: hand_num (hand id number to detect, default is zero)
        :draw: bool (draws circles over the hand keypoints, default is true)

        :returns: 
            lm_list (list of lists of keypoints)
            img
        '''
        lm_list = []
        h, w, c = img.shape

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_num]
            for id_point, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id_point, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

        return lm_list, img

    def findHand3DPosition(self, hand_num=0, draw=False):
        '''
        Find the hand 3d positions on the referred detected hand in real-world 3D coordinates 
        that are in meters with the origin at the hand's approximate geometric center.
        Please refer to the documentation for further details: 
        https://google.github.io/mediapipe/solutions/hands.html#multi_hand_world_landmarks


        :param: hand_num (hand id number to detect, default is zero)
        :draw: bool (draws a 3d graph of the predicted locations in world coordinates of the hand keypoints, default is False)

        :returns: list of lists of 3d hand keypoints in the format [[id_point, x_point,y_point,z_point]]
        '''
        lm3d_list = []
        if self.results.multi_hand_world_landmarks:
            hand3d = self.results.multi_hand_world_landmarks[hand_num]
            for id_point, lm in enumerate(hand3d.landmark):
                lm3d_list.append([id_point, lm.x, lm.y, lm.z])
            if draw:
                self.mpDraw.plot_landmarks(
                    hand3d, self.mpHands.HAND_CONNECTIONS, azimuth=5)
        return lm3d_list

    # TO DO: fix pose estimation function...
    def findHandPose(self, lmlist, lm3dlist, frame, camera_matrix=None, dist_coeffs=None, draw_axis=True, axis_scale=0.05):
        '''
        Estimate hand pose using the 2d and 3d keypoints of the hand.

        Parameters
        ----------
        frame: opencv image array
            contains frame to be processed
        lmlist:
            list of 2d landmarks detected of the 
        lm3dlist:
            list of 3d landmarks detected of the hand, in world coordinates
        draw_axis: bool
            If set to True, shows the head pose axis projected from the keypoints
            used for pose estimation (default is True)

        Returns
        --------
        - if successful: image_frame, roll, pitch, yaw (tuple)
        - if unsuccessful: None,None,None,None (tuple)

        '''

        self.hand_lm = lmlist
        self.hand_3dlm = lm3dlist
        self.frame = frame  # opencv image array

        self.axis = np.float32([[axis_scale, 0, 0],
                                [0, axis_scale, 0],
                                [0, 0, axis_scale]])
        # array that specify the length of the 3 projected axis from the nose

        if camera_matrix is None:
            # if no camera matrix is given, estimate camera parameters using picture size
            self.size = frame.shape
            self.focal_length = self.size[1]
            self.center = (self.size[1] / 2, self.size[0] / 2)
            self.camera_matrix = np.array(
                [[self.focal_length, 0, self.center[0]],
                 [0, self.focal_length, self.center[1]],
                 [0, 0, 1]], dtype="double"
            )
        else:
            # take camera matrix
            self.camera_matrix = camera_matrix

        if dist_coeffs is None:  # if no distorsion coefficients are given, assume no lens distortion
            self.dist_coeffs = np.zeros((4, 1))
        else:
            # take camera distortion coefficients
            self.dist_coeffs = dist_coeffs

        # index,middle finger, ring finger, pinky and thumb keypoints in the image
        self.thumb_mcp_lm = tuple(self.hand_lm[1][1:])
        self.index_finger_mcp_lm = tuple(self.hand_lm[5][1:])
        self.middle_finger_mcp_lm = tuple(self.hand_lm[9][1:])
        self.ring_finger_mcp_lm = tuple(self.hand_lm[13][1:])
        self.pinky_mcp_lm = tuple(self.hand_lm[17][1:])
        self.wrist_lm = tuple(self.hand_lm[0][1:])

        # index,middle finger, ring finger, pinky and thumb 3d
        # estimated position in world space coordinates

        self.thumb_mcp_3dlm = tuple(self.hand_3dlm[1][1:])
        self.index_finger_mcp_3dlm = tuple(self.hand_3dlm[5][1:])
        self.middle_finger_mcp_3dlm = tuple(self.hand_3dlm[9][1:])
        self.ring_finger_mcp_3dlm = tuple(self.hand_3dlm[13][1:])
        self.pinky_mcp_3dlm = tuple(self.hand_3dlm[17][1:])
        self.wrist_3dlm = tuple(self.hand_3dlm[0][1:])

        # 3D hand keypoints in world space coordinates
        self.model_points = np.array([
            self.thumb_mcp_3dlm,
            self.index_finger_mcp_3dlm,
            self.middle_finger_mcp_3dlm,
            self.ring_finger_mcp_3dlm,
            self.pinky_mcp_3dlm,
            self.wrist_3dlm,
        ], dtype="double")

        # 2D hand keypoints position in the image (frame)
        self.image_points = np.array([
            self.thumb_mcp_lm,
            self.index_finger_mcp_lm,
            self.middle_finger_mcp_lm,
            self.ring_finger_mcp_lm,
            self.pinky_mcp_lm,
            self.wrist_lm,
        ], dtype="double")

        self.draw = draw_axis

        (success, rvec, tvec) = cv2.solvePnP(self.model_points, self.image_points,
                                             self.camera_matrix, self.dist_coeffs)

        if success:  # if the solvePnP succeed, compute the head pose, otherwise return None

            rvec, tvec = cv2.solvePnPRefineVVS(
                self.model_points, self.image_points, self.camera_matrix, self.dist_coeffs, rvec, tvec)
            # this method is used to refine the rvec and tvec prediction

            # knuckle (index_finger_mcp_lm) point on the image plane
            knuckle = (int(self.image_points[1][0]), int(
                self.image_points[1][1]))

            (end_point2D, _) = cv2.projectPoints(
                self.axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            Rmat = cv2.Rodrigues(rvec)[0]
            # using the Rodrigues formula, this functions computes the Rotation Matrix from the rotation vector
            P = np.hstack((Rmat, tvec))  # computing the projection matrix

            euler_angles = cv2.decomposeProjectionMatrix(P)[6]
            yaw, pitch, roll = euler_angles[0][0], euler_angles[1][0], euler_angles[2][0]

        if self.draw:
            self.frame = draw_pose_info(
                self.frame, knuckle, end_point2D, yaw, pitch, roll)
            # draws 3d axis from the nose and to the computed projection points
            for point in self.image_points:
                cv2.circle(self.frame, tuple(
                    point.ravel().astype(int)), 2, (0, 255, 255), -1)
            # draws the 6 keypoints used for the pose estimation
            return self.frame, yaw, pitch, roll

        else:
            return self.frame, None, None, None


# MAIN SCRIPT EXAMPLE FOR REAL-TIME HAND TRACKING USING A WEBCAM
def main(camera_source=0, show_fps=True, verbose=False):

    # camera calibration parameters (example)
    camera_matrix = np.array(
        [[899.12150372, 0., 644.26261492],
         [0., 899.45280671, 372.28009436],
            [0, 0,  1]], dtype="double")

    dist_coeffs = np.array(
        [[-0.03792548, 0.09233237, 0.00419088, 0.00317323, -0.15804257]], dtype="double")

    ctime = 0  # current time (used to compute FPS)
    ptime = 0  # past time (used to compute FPS)

    cv2.setUseOptimized(True)

    # capture the input from the default system camera (camera number 0)
    cap = cv2.VideoCapture(camera_source)
    detector = HandDetector()

    if not cap.isOpened():  # if the camera can't be opened exit the program
        print("Cannot open camera")
        exit()

    while True:  # infinite loop for webcam video capture

        ret, frame = cap.read()  # read a frame from the webcam

        if not ret:  # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break

        frame = detector.findHands(frame)
        hand_lmlist, frame = detector.findHandPosition(
            frame, hand_num=0, draw=False)
        hand_3dlmlist = detector.findHand3DPosition()

        if len(hand_lmlist) > 0 and len(hand_3dlmlist) > 0:
            frame, yaw, pitch, roll = detector.findHandPose(
                lmlist=hand_lmlist, lm3dlist=hand_3dlmlist, frame=frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
            if verbose:
                print(
                    f"hand keypoints:\n{hand_lmlist}\nhand 3d keypoints position:\n{hand_3dlmlist}")

        # compute the actual frame rate per second (FPS) of the webcam video capture stream, and show it
        ctime = time.time()
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
    main(camera_source=0)
