import mediapipe as mp
import numpy as np
import cv2

import time


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

    def findHands(self, frame, draw=True):
        '''
        Detects the hands and draws keypoints of the hands given and input image.
        :param: img (opencv image in BGR)
        :param: draw (boolean, draw the keypoint if set to true, default is true)
        :returns: img (opencv image in BGR with keypoints drawn if draw is set to true)
        '''
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLMs in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLMs,
                                               self.mpHands.HAND_CONNECTIONS)
        return frame

    def findHandPosition(self, frame, hand_num=0, draw=True):
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
        self.lm_list = []
        h, w, c = frame.shape

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_num]
            for id_point, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id_point, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        return self.lm_list, frame

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
        self.lm3d_list = []
        if self.results.multi_hand_world_landmarks:
            hand3d = self.results.multi_hand_world_landmarks[hand_num]
            for id_point, lm in enumerate(hand3d.landmark):
                self.lm3d_list.append([id_point, lm.x, lm.y, lm.z])
            if draw:
                self.mpDraw.plot_landmarks(
                    hand3d, self.mpHands.HAND_CONNECTIONS, azimuth=5)
        return self.lm3d_list

    # TO DO: fix pose estimation function...
    def findHandPose(self, frame, camera_matrix=None, dist_coeffs=None, draw_axis=True, axis_scale=2):
        '''
        Estimate hand pose using the 2d and 3d keypoints of the hand.

        Parameters
        ----------
        frame: opencv image array
            contains frame to be processed
        draw_axis: bool
            If set to True, shows the head pose axis projected from the keypoints
            used for pose estimation (default is True)
        camera_matrix: np.array 
            matrix of the camera parameters (default is None)
        dist_coeffs: np.array
            array of distorsion coefficients of the camera (default is None)

        Returns
        --------
        - if successful: image_frame, yaw, pitch, roll (tuple)
        - if unsuccessful: None,None,None,None (tuple)

        '''

        self.hand_lm = self.lm_list
        self.hand_3dlm = self.lm3d_list

        self.axis = np.float32([[axis_scale, 0, 0],
                                [0, axis_scale, 0],
                                [0, 0, axis_scale]])
        # array that specify the length of the 3 projected axis from the nose

        if camera_matrix is None:
            # if no camera matrix is given, estimate camera parameters using picture size
            size = frame.shape
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
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

        # convert 3d points in meters to centimeters
        for i, lm3d in enumerate(self.hand_3dlm):
            self.hand_3dlm[i] = [i, lm3d[1:][0]
                                 * 100, lm3d[1:][1] * 100, lm3d[1:][2] * 100]
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
            self.wrist_3dlm
        ], dtype="double")

        # 2D hand keypoints position in the image (frame)
        self.image_points = np.array([
            self.thumb_mcp_lm,
            self.index_finger_mcp_lm,
            self.middle_finger_mcp_lm,
            self.ring_finger_mcp_lm,
            self.pinky_mcp_lm,
            self.wrist_lm
        ], dtype="double")

        (success, rvec, tvec) = cv2.solvePnP(self.model_points, self.image_points,
                                             self.camera_matrix, self.dist_coeffs,
                                             flags=cv2.SOLVEPNP_ITERATIVE)

        if success:  # if the solvePnP succeed, compute the head pose, otherwise return None

            rvec, tvec = cv2.solvePnPRefineVVS(
                self.model_points, self.image_points, self.camera_matrix, self.dist_coeffs, rvec, tvec)
            # this method is used to refine the rvec and tvec prediction

            # middle palm point
            middle_palm_point = (int((self.image_points[1][0] + self.image_points[4][0] + self.image_points[5][0])/3), int(
                (self.image_points[1][1] + self.image_points[4][1] + self.image_points[5][1])/3))

            (end_point2D, _) = cv2.projectPoints(
                self.axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            Rmat = cv2.Rodrigues(rvec)[0]
            # using the Rodrigues formula, this functions computes the Rotation Matrix from the rotation vector
            P = np.hstack((Rmat, tvec))  # computing the projection matrix

            euler_angles = cv2.decomposeProjectionMatrix(P)[6]
            yaw, pitch, roll = euler_angles[0][0], euler_angles[1][0], euler_angles[2][0]

        if draw_axis:
            frame = self.draw_pose_info(
                frame, middle_palm_point, end_point2D, yaw=yaw, pitch=pitch, roll=roll)
            # draws 3d axis from the nose and to the computed projection points
            for point in self.image_points:
                cv2.circle(frame, tuple(
                    point.ravel().astype(int)), 2, (0, 255, 255), -1)
            # draws the 6 keypoints used for the pose estimation
            return frame, yaw, pitch, roll

        else:
            return frame, None, None, None

    def findHandAperture(self, frame, verbose=False, show_aperture=True):
        '''
        Finds the normalized hand aperture as distance between the mean point of the hand tips and the mean wrist and thumb base point divided by the palm lenght.

        Parameters
        ----------
        frame: opencv image array
            contains frame to be processed
        verbose: bool
            If set to True, prints the hand aperture value on the frame (default is False)
        show_aperture: bool
            If set to True, show the hand aperture with a line

        Returns
        --------
        frame, hand aperture (aperture)
        In case the aperture can't be computed, the value of aperture will be None
        '''
        aperture = None

        thumb_cmc_lm_array = np.array([self.lm_list[1][1:]])[0]
        wrist_lm_array = np.array([self.lm_list[0][1:]])[0]
        lower_palm_midpoint_array = (thumb_cmc_lm_array + wrist_lm_array) / 2

        index_mcp_lm_array = np.array([self.lm_list[5][1:]])[0]
        pinky_mcp_lm_array = np.array([self.lm_list[5][1:]])[0]
        upper_palm_midpoint_array = (
            index_mcp_lm_array + pinky_mcp_lm_array) / 2

        # compute palm size as l2 norm between the upper palm midpoint and lower palm midpoint
        palm_size = np.linalg.norm(
            upper_palm_midpoint_array - lower_palm_midpoint_array, ord=2)
        # print(f"palm size:{palm_size}")

        index_tip_array = np.array([self.lm_list[8][1:]])[0]
        middle_tip_array = np.array([self.lm_list[12][1:]])[0]
        ring_tip_array = np.array([self.lm_list[16][1:]])[0]
        pinky_tip_array = np.array([self.lm_list[20][1:]])[0]

        hand_tips = np.array([index_tip_array,
                              middle_tip_array,
                              ring_tip_array,
                              pinky_tip_array])
        # print(f"hand_tips: {hand_tips}")

        tips_midpoint_array = np.mean(hand_tips, axis=0)
        # print(f"tips_midpoint_array:{tips_midpoint_array}")

        # compute hand aperture as l2norm between hand tips midpoint and lower palm midpoint
        # normalize by palm size computed before
        aperture = np.round(np.linalg.norm(
            tips_midpoint_array - lower_palm_midpoint_array, ord=2)/palm_size, 3)

        if verbose:
            cv2.putText(frame, "HAND APERTURE:" + str(aperture), (10, 40),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
        if show_aperture:
            frame = cv2.line(frame, tuple(tips_midpoint_array.astype(int)),
                             tuple(lower_palm_midpoint_array.astype(int)), (255, 0, 0), 3)

        return frame, aperture

    @staticmethod
    def draw_pose_info(frame, img_point, point_proj, roll=None, pitch=None, yaw=None):
        """
        Draw 3d orthogonal axis given a frame, a point in the frame, the projection point array.
        Also prints the information about the roll, pitch and yaw if passed

        :param frame: opencv image/frame
        :param img_point: tuple
            x,y position in the image/frame for the 3d axis for the projection
        :param point_proj: np.array
            Projected point along 3 axis obtained from the cv2.projectPoints function
        :param roll: float, optional
        :param pitch: float, optional
        :param yaw: float, optional
        :return: frame: opencv image/frame
            Frame with 3d axis drawn and, optionally, the roll,pitch and yaw values drawn
        """
        frame = cv2.line(frame, img_point, tuple(
            point_proj[0].ravel().astype(int)), (255, 0, 0), 3)
        frame = cv2.line(frame, img_point, tuple(
            point_proj[1].ravel().astype(int)), (0, 255, 0), 3)
        frame = cv2.line(frame, img_point, tuple(
            point_proj[2].ravel().astype(int)), (0, 0, 255), 3)

        if roll is not None and pitch is not None and yaw is not None:
            cv2.putText(frame, "Roll:" + str(np.round(roll, 1)), (500, 50),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Pitch:" + str(np.round(pitch, 1)), (500, 70),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Yaw:" + str(np.round(yaw, 1)), (500, 90),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

        return frame


# ---------------------------------------------------------------
# MAIN SCRIPT EXAMPLE FOR REAL-TIME HAND TRACKING USING A WEBCAM
# ---------------------------------------------------------------

def main(camera_source=0, show_fps=True, verbose=False):

    assert camera_source >= 0, f"source needs to be greater or equal than 0\n"

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
    detector = HandDetector(detCon=0.7, trackCon=0.7)

    if not cap.isOpened():  # if the camera can't be opened exit the program
        print("Cannot open camera")
        exit()

    while True:  # infinite loop for webcam video capture

        ret, frame = cap.read()  # read a frame from the webcam

        if not ret:  # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break

        frame = detector.findHands(frame=frame)
        hand_lmlist, frame = detector.findHandPosition(
            frame=frame, hand_num=0, draw=False)
        # hand_3dlmlist = detector.findHand3DPosition()

        """ if len(hand_lmlist) > 0 and len(hand_3dlmlist) > 0:
            frame, yaw, pitch, roll = detector.findHandPose(
                lmlist=hand_lmlist, lm3dlist=hand_3dlmlist, frame=frame)
            if verbose:
                print(
                    f"hand keypoints:\n{hand_lmlist}\nhand 3d keypoints position:\n{hand_3dlmlist}") """

        if len(hand_lmlist) > 0:
            frame, aperture = detector.findHandAperture(
                frame=frame, verbose=True, show_aperture=True)

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
