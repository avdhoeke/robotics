import cv2
import numpy as np
from src import Network


class Agent:

    def __init__(self):
        self.network = Network()

    def run(self):
        while True:

            self.network.send("Hello!")
            data = self.network.recv()

            # OpenCV function
            WINDOW_NAME = "Robotics Project"
            cv2.namedWindow(WINDOW_NAME)  # open a window to show debugging images
            vc = cv2.VideoCapture(0)  # Initialize the default camera

            try:
                if vc.isOpened():  # try to get the first frame
                    (readSuccessful, frame) = vc.read()
                else:
                    raise (Exception("failed to open camera."))

                while readSuccessful:
                    # Export image in HSV format
                    _ = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    # lower boundary RED color range values; Hue (0 - 10)
                    lower1 = np.array([0, 0, 217])
                    upper1 = np.array([10, 255, 255])

                    # upper boundary RED color range values; Hue (160 - 180)
                    lower2 = np.array([90, 0, 230])
                    upper2 = np.array([179, 255, 255])

                    lower_mask = cv2.inRange(_, lower1, upper1)
                    upper_mask = cv2.inRange(_, lower2, upper2)

                    full_mask = lower_mask + upper_mask

                    total = cv2.bitwise_and(frame, frame, mask=full_mask)

                    # Additional Red color filter
                    low_red = np.array([10, 0, 0])
                    high_red = np.array([180, 255, 255])
                    red_mask = cv2.inRange(total, low_red, high_red)
                    red = cv2.bitwise_and(frame, frame, mask=red_mask)

                    # Treshold the resulting image
                    h, s, v = cv2.split(red)
                    ret, final = cv2.threshold(v, 250, 255, cv2.THRESH_BINARY)

                    # Dilatation
                    dilatation = cv2.dilate(final, np.ones((3, 3)))
                    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(dilatation)

                    # Compute position of biggest red area
                    x, y = None, None
                    max_area = None

                    for stat, center in zip(stats[1:], centroids[1:]):
                        area = stat[4]

                        if (max_area is None) or (area > max_area):
                            x, y = center
                            max_area = area

                    image = np.copy(frame)
                    if x is not None and y is not None:
                        x, y = int(x), int(y)
                        image[y - 10:y + 10, x - 10:x + 10, :] = (100, 100, 255)

                    # Display the resulting filtered image
                    cv2.imshow('Image with mask', total)
                    cv2.imshow('Filtered image', final)
                    cv2.imshow('Filtered image with contour', image)
                    cv2.imshow('Filtered image with red dominance', red)

                    # Set refreshing time
                    key = cv2.waitKey(10)
                    if key == 27: # exit on ESC
                        break
                    # Get Image from camera
                    readSuccessful, frame = vc.read()
            finally:
                vc.release()  # close the camera
                cv2.destroyWindow(WINDOW_NAME)  # close the window
