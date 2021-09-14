import cv2
import numpy as np


def to_grayscale_and_resize(image, width = 64, height = 64):
    img = image.copy()
    # First axis of doom screen buffer is the number of channels, while the OpenCV takes channels as the last parameter.
    img = np.moveaxis(img, 0, -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    # Need to reshape again to get it in the way (1, height, width)
    img = img.reshape(1, img.shape[0], img.shape[1])
    return img


def to_resize(image, width = 64, height = 64):
    img = image.copy()
    # First axis of doom screen buffer is the number of channels, while the OpenCV takes channels as the last parameter.
    img = cv2.resize(img, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    # Need to reshape again to get it in the way (1, height, width)
    img = img.reshape(1, img.shape[0], img.shape[1])
    return img