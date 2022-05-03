import cv2
import time


def videocapture(fps_cap=60, show_fps=True, source=0):
    """
    Capture webcam video from the specified "source" (default is 0) using the opencv VideoCapture function.
    It's possible to cap/limit the number of FPS using the "fps_cap" variable (default is 60) and to show the actual FPS footage (shown by default).
    The program stops if "q" is pressed or there is an error in opening/using the capture source.

    :param: fps_cap - max framerate allowed (default is 60)
    :param: show_fps - shows a real-time framerate indicator (default is True)
    :param: source - select the webcam source number used in OpenCV (default is 0)

    """

    ctime = 0  # current time (used to compute FPS)
    ptime = 0  # past time (used to compute FPS)
    prev_time = 0  # previous time variable, used to set the FPS limit

    fps_lim = fps_cap  # FPS upper limit value, needed for estimating the time for each frame and increasing performances

    time_lim = 1. / fps_lim  # time window for each frame taken by the webcam

    # capture the input from the default system camera (camera number 0)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():  # if the camera can't be opened exit the program
        print("Cannot open camera")
        exit()

    while True:  # infinite loop for webcam video capture

        delta_time = time.time() - prev_time  # computed  delta time for FPS capping

        ret, frame = cap.read()  # read a frame from the webcam

        if not ret:  # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break

        if delta_time >= time_lim:  # if the time passed is bigger or equal than the frame time, process the frame
            prev_time = time.time()

            # compute the actual frame rate per second (FPS) of the webcam video capture stream, and show it
            ctime = time.time()
            fps = 1.0 / float(ctime - ptime)
            ptime = ctime

            if show_fps:
                cv2.putText(frame, "FPS:" + str(round(fps, 0)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2,
                            (255, 255, 255), 1)

            # IMAGE PROCESSING CODE HERE

            # show the frame on screen
            cv2.imshow("Frame (press 'q' to exit)", frame)

        # if the key "q" is pressed on the keyboard, the program is terminated
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    videocapture()
