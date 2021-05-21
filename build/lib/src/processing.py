import cv2
import numpy as np
from typing import *
from . import RaspEnv


def get_red_dot(env: RaspEnv, display_images: bool) -> None:

    # Get attributes of environment instance
    frame = env.frame

    # Process the input image from webcam
    total, red, final = filter_image(frame)

    # Place our red dot on image
    x, y, image = compute_red_dot(final, frame)

    if display_images:
        # Display the resulting filtered images
        cv2.imshow('Image filtered with mask', total)
        cv2.imshow('Filtered image with red dominance', red)
        cv2.imshow('Filtered image with binary threshold', final)

    # Display final image with red square on it
    cv2.imshow('Original image with red square', image)

    # Set new square location
    env.square_ = [x, y]


def filter_image(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Convert BGR to HSV format
    _ = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 0, 217])
    upper1 = np.array([10, 255, 255])

    # upper boundary RED color range values; Hue (90 - 180)
    lower2 = np.array([90, 0, 230])
    upper2 = np.array([179, 255, 255])

    # Apply filters defined previously
    lower_mask = cv2.inRange(_, lower1, upper1)
    upper_mask = cv2.inRange(_, lower2, upper2)

    full_mask = lower_mask + upper_mask

    total = cv2.bitwise_and(frame, frame, mask=full_mask)

    # Additional Red color filter
    low_red = np.array([10, 0, 0])
    high_red = np.array([180, 150, 255])
    red_mask = cv2.inRange(total, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Threshold the resulting image
    h, s, v = cv2.split(red)
    ret, final = cv2.threshold(v, 150, 255, cv2.THRESH_BINARY)

    return total, red, final


def compute_red_dot(final: np.ndarray, frame: np.ndarray) -> Tuple[Union[None, float], Union[None, float], np.ndarray]:
    # Dilatation of filtered image
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

    # Put blue square at the center of the image
    image[360 - 10:360 + 10, 640 - 10:640 + 10, :] = (255, 100, 100)

    if x is not None and y is not None:
        x, y = int(x), int(y)
        image[y - 10:y + 10, x - 10:x + 10, :] = (100, 100, 255)
        return float(x), float(y), image

    else:
        return x, y, image
