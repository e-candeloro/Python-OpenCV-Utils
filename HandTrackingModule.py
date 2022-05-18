import mediapipe as mp
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

        :returns: lm_list (list of lists of keypoints)
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

        return lm_list

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


# MAIN SCRIPT EXAMPLE FOR REAL-TIME HAND TRACKING USING A WEBCAM
def main(camera_source=0, show_fps=True):
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
        hand_lmlist = detector.findHandPosition(frame, hand_num=0, draw=False)
        hand_3dlmlist = detector.findHand3DPosition()

        if hand_lmlist != 0 and hand_3dlmlist != 0:
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
