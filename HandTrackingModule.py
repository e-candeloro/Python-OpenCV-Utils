import mediapipe as mp
import cv2
import time

class HandDetector():
    def __init__(self, mode=False, maxHands = 1,modCompl = 1, detCon = 0.5, trackCon = 0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.modCompl = modCompl
        self.detCon = detCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
               max_num_hands=self.maxHands,
               model_complexity=self.modCompl,
               min_detection_confidence=self.detCon,
               min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.results = None

    def findHands(self, img, draw = True):
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

    def findHandPosition(self, img, hand_num = 0, draw = True):
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


def main(camera_source=0, show_fps=True):

    ctime = 0  # current time (used to compute FPS)
    ptime = 0  # past time (used to compute FPS)

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
        if len(hand_lmlist) > 0:
            print(hand_lmlist)

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
