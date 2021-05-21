import cv2
import numpy as np


def run():
    while True:

        def nothing(x):
            pass

        # Create a window
        cv2.namedWindow('image')

        # create trackbars for color change
        cv2.createTrackbar('HMin', 'image', 0, 179, nothing)  # Hue is from 0-179 for Opencv
        cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
        cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
        cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
        cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
        cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

        # Set default value for MAX HSV trackbars.
        cv2.setTrackbarPos('HMax', 'image', 179)
        cv2.setTrackbarPos('SMax', 'image', 255)
        cv2.setTrackbarPos('VMax', 'image', 255)

        # Initialize to check if HSV min/max value changes
        hMin = sMin = vMin = hMax = sMax = vMax = 0
        phMin = psMin = pvMin = phMax = psMax = pvMax = 0

        # OpenCV function
        WINDOW_NAME = "Calibration of filter"
        cv2.namedWindow(WINDOW_NAME)  # open a window to show debugging images
        vc = cv2.VideoCapture(0)  # Initialize the default camera

        try:
            if vc.isOpened():  # try to get the first frame
                (readSuccessful, frame) = vc.read()
            else:
                raise (Exception("failed to open camera."))

            while readSuccessful:

                # get current positions of all trackbars
                hMin = cv2.getTrackbarPos('HMin', 'image')
                sMin = cv2.getTrackbarPos('SMin', 'image')
                vMin = cv2.getTrackbarPos('VMin', 'image')

                hMax = cv2.getTrackbarPos('HMax', 'image')
                sMax = cv2.getTrackbarPos('SMax', 'image')
                vMax = cv2.getTrackbarPos('VMax', 'image')

                # Set minimum and max HSV values to display
                lower = np.array([hMin, sMin, vMin])
                upper = np.array([hMax, sMax, vMax])

                # Create HSV Image and threshold into a range.
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower, upper)
                output = cv2.bitwise_and(frame, frame, mask=mask)

                # Print if there is a change in HSV value
                if ((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (
                        pvMax != vMax)):
                    print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (
                        hMin, sMin, vMin, hMax, sMax, vMax))
                    phMin = hMin
                    psMin = sMin
                    pvMin = vMin
                    phMax = hMax
                    psMax = sMax
                    pvMax = vMax

                # Display output image
                cv2.imshow('image', output)
                #############################

                # Set refreshing time
                key = cv2.waitKey(10)
                if key == 27:  # exit on ESC
                    break
                # Get Image from camera
                readSuccessful, frame = vc.read()
        finally:
            vc.release()  # close the camera
            cv2.destroyWindow(WINDOW_NAME)  # close the window


run()
