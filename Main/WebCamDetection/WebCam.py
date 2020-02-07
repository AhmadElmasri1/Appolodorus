import pyrealsense2 as rs
import venv
import cv2 as cv
import numpy as np
import Camera
from matplotlib import pyplot as plt

mainCamera = Camera.CameraHandler()
mainCamera.start()

while True:
    colorImage = mainCamera.getColorFrame()
    depthImage = mainCamera.getDepthFrame()

    # colorImage = np.float32(colorImage)

    images = np.hstack((colorImage, depthImage))
    # referenceImage =
    # referenceImage = np.uint8(referenceImage.copy())


    # plt.imshow(displayImage)
    # plt.show()

    cv.namedWindow('Cam Output', cv.WINDOW_AUTOSIZE)
    cv.imshow('Cam Output', images)
    # cv.waitKey(1)