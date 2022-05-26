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
        If additional info is True, returns a list of list in the format
        [[id_point0, x_point0, y_point0, zpoint0, visibility], ...]

        Keypoints list  are shown on this site: https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png

        :param: additional_info (returns z and visibility in the keypoint list. Default is False)
        :param: frame(opencv BGR image)
        :draw: bool (draws circles over the keypoints. Default is True)

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

    def find3DPosePosition(self, additional_info=False, draw=False):
        '''
        Given and image, returns the 3D pose keypoints position in the format of a list of lists
        [[id_point0, x_point0, y_point0, zpoint0], ...]
        The keypoints are in world coordinates and in meter, with the origin in the middle point of the hips
        If additional info is True, returns a list of list in the format
        [[id_point0, x_point0, y_point0, zpoint0, visibility], ...]

        Keypoints list  are shown on this site: https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png

        :param: additional_info (returns visibility in the keypoint list. Default is False)
        :draw: bool (draws a matplotlib 3d graph of all the keypoints in world coordinates. Default is False)

        :returns: 
            lm_3dlist (list of lists of keypoints)
        '''
        self.lm_3dlist = []

        if self.results.pose_landmarks:
            pose = self.results.pose_world_landmarks
            for id_point, lm in enumerate(pose.landmark):
                cx, cy, cz = lm.x, lm.y, lm.z

                if additional_info:
                    vis = lm.visibility
                    self.lm_3dlist.append([id_point, cx, cy, cz, vis])
                else:
                    self.lm_3dlist.append([id_point, cx, cy, cz])

            if draw:
                self.mpDraw.plot_landmarks(
                    self.results.pose_world_landmarks, self.mpPose.POSE_CONNECTIONS)

        return self.lm_3dlist

    def findAngle(self, frame, p1: int, p2: int, p3: int, angle3d=False, draw=True):
        '''Find the angle between 3 points p1, p2, p3 in succession, where p2 is the point where the angle is measured.
        For the points, only the index number is required. Please refer to this image to select the appriopriate keypoints: https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png

        Example: elbow angle, given the shoulder keypoint, the elbow keypoint and the wrist keypoint

        :param: frame (opencv frame)
        :p1:first point index
        :p2:second point index
        :p3:third point index
        :angle3d: Bool: performs 3d angle computation, default is False
        :flip_2dangle: Bool: flips the angle computation if it is in 2d, default is False
        :draw: Bool (optional): draws additional info, default is True

        Returns:
            -angle: angle in degrees between the segment s12 and the segment s23 having p2 as vertex, where the angle is located
        '''
        # checks if keypoints values are correct
        assert p1 >= 0 and p1 <= 32, f"p1 must be >=0 and <=32"
        assert p2 >= 0 and p2 <= 32, f"p2 must be >=0 and <=32"
        assert p3 >= 0 and p3 <= 32, f"p3 must be >=0 and <=32"
        assert len(
            self.lm_list) > 0, f"Landmark list is empty, use this function only after using the FindPose and FindPosePosition methods"
        assert len(
            self.lm_3dlist) > 0, f"3D Landmark list is empty, use this function only after using the FindPose and Find3DPosePosition methods"

        if angle3d:

            x1, y1, z1 = self.lm_3dlist[p1][1:4]
            x2, y2, z2 = self.lm_3dlist[p2][1:4]
            x3, y3, z3 = self.lm_3dlist[p3][1:4]

            v21 = np.array([x1 - x2, y1 - y2, z1 - z2]) * 100
            v32 = np.array([x3 - x2, y3 - y2, z3 - z2]) * 100

        else:

            x1, y1 = self.lm_list[p1][1:3]
            x2, y2 = self.lm_list[p2][1:3]
            x3, y3 = self.lm_list[p3][1:3]

            v21 = np.array([x1 - x2, y1 - y2]) * 100
            v32 = np.array([x3 - x2, y3 - y2]) * 100

        angle = np.degrees(
            np.arccos(
                np.dot(v21, v32)/(np.linalg.norm(v21, 2)
                                  * np.linalg.norm(v32, 2))
            )
        )

        if draw:
            cx, cy = self.lm_list[p2][1:3]
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 10, (255, 0, 255), 1)
            cv2.putText(frame, str(round(angle, 0)), (cx - 50, cy + 50),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return angle

    # TODO: improve pose estimation function

    def findBody3DPose(self, frame, camera_matrix=None, dist_coeffs=None, draw_axis=True, axis_scale=2):
        '''
        Estimate body torso pose (yaw, pitch roll) using the 2d and 3d keypoints.

        Parameters
        ----------
        frame: opencv image array
            contains frame to be processed
        draw_axis: bool
            If set to True, shows the pose axis projected from the keypoints
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

        self.pose_lm = self.lm_list
        self.pose_3dlm = self.lm_3dlist

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
        for i, lm3d in enumerate(self.pose_3dlm):
            self.pose_3dlm[i] = [i, lm3d[1:][0]
                                 * 100, lm3d[1:][1] * 100, lm3d[1:][2] * 100]

        # left and right shoulders, left and right hips, left and right knee 2d keypoints
        self.left_mouth_lm = tuple(self.pose_lm[9][1:3])
        self.right_mouth_lm = tuple(self.pose_lm[10][1:3])
        self.left_shoulder_lm = tuple(self.pose_lm[11][1:3])
        self.right_shoulder_lm = tuple(self.pose_lm[12][1:3])
        self.left_hip_lm = tuple(self.pose_lm[23][1:3])
        self.right_hip_lm = tuple(self.pose_lm[24][1:3])

        #  # left and right shoulders, left and right hips, left and right knee 3d keypoints
        # estimated position in world space coordinates

        self.left_mouth_3dlm = tuple(self.pose_3dlm[9][1:4])
        self.right_mouth_3dlm = tuple(self.pose_3dlm[10][1:4])
        self.left_shoulder_3dlm = tuple(self.pose_3dlm[11][1:4])
        self.right_shoulder_3dlm = tuple(self.pose_3dlm[12][1:4])
        self.left_hip_3dlm = tuple(self.pose_3dlm[23][1:4])
        self.right_hip_3dlm = tuple(self.pose_3dlm[24][1:4])

        # 3D hand keypoints in world space coordinates
        self.model_points = np.array([
            self.left_mouth_3dlm,
            self.right_mouth_3dlm,
            self.left_shoulder_3dlm,
            self.right_shoulder_3dlm,
            self.left_hip_3dlm,
            self.right_hip_3dlm,

        ], dtype="double")

        # 2D keypoints position in the image (frame)
        self.image_points = np.array([
            self.left_mouth_lm,
            self.right_mouth_lm,
            self.left_shoulder_lm,
            self.right_shoulder_lm,
            self.left_hip_lm,
            self.right_hip_lm,

        ], dtype="double")

        (success, rvec, tvec) = cv2.solvePnP(self.model_points, self.image_points,
                                             self.camera_matrix, self.dist_coeffs,
                                             flags=cv2.SOLVEPNP_ITERATIVE)

        if success:  # if the solvePnP succeed, compute the head pose, otherwise return None

            rvec, tvec = cv2.solvePnPRefineVVS(
                self.model_points, self.image_points, self.camera_matrix, self.dist_coeffs, rvec, tvec)
            # this method is used to refine the rvec and tvec prediction

            # middle shoulder point
            middle_shoulder_point = (int((self.image_points[2][0] + self.image_points[3][0])/2), int(
                (self.image_points[2][1] + self.image_points[3][1])/2))

            (end_point2D, _) = cv2.projectPoints(
                self.axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            Rmat = cv2.Rodrigues(rvec)[0]
            # using the Rodrigues formula, this functions computes the Rotation Matrix from the rotation vector
            P = np.hstack((Rmat, tvec))  # computing the projection matrix

            euler_angles = cv2.decomposeProjectionMatrix(P)[6]
            yaw, pitch, roll = euler_angles[0][0], euler_angles[1][0], euler_angles[2][0]

        if draw_axis:
            frame = self.draw_pose_info(
                frame, middle_shoulder_point, end_point2D, yaw=yaw, pitch=pitch, roll=roll)
            # draws 3d axis from the nose and to the computed projection points
            for point in self.image_points:
                cv2.circle(frame, tuple(
                    point.ravel().astype(int)), 2, (0, 255, 255), -1)
            # draws the 6 keypoints used for the pose estimation
            return frame, yaw, pitch, roll

        else:
            return frame, None, None, None


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
    detector = poseDetector(detCon=0.7, trackCon=0.7, modCompl=1)

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
        lm_3dlist = detector.find3DPosePosition()

        if len(lm_list) > 0 and len(lm_3dlist) > 0:
            """ angle = detector.findAngle(
                frame, 12, 14, 16, flip_angle=True, draw=True) """

            angle_3d = detector.findAngle(
                frame, 12, 14, 16, angle3d=True, draw=True)
            """ frame, yaw, pitch, roll = detector.findBody3DPose(
                frame, draw_axis=True, axis_scale=2) """

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
    main(camera_source=0)
