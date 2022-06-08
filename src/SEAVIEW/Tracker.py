"""Contains fish tracking support classes

Credit: Unless otherwise stated, code by Rose Awen Brindle
"""

import cv2
import numpy as np

import ImageHelpers

low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])

class Tracker():

    def __init__(self, vidname, nmarkers, calib):
        """
        """
        self.calib = calib

        self.frames = self.loadVideo(vidname)

        self.data = []

    def anaylse(self):
        # TODO preprocess img to make circles as visible as possible
        # TODO identify circles
        # TODO select circles most likely to be markers on fish robot

        for im in self.frames:
            ims_out = [] 

            ims_out.append(im.copy())

            # basd on code by robert
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            hsv = cv2.medianBlur(hsv, 9)
            
            blur_ex = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            ims_out.append(blur_ex)

            green_mask = cv2.inRange(hsv, low_green, high_green)
            green = cv2.bitwise_and(im, im, mask=green_mask) 

            ims_out.append(green)

            # get user input
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                # quit early
                break
            elif k == ord('p'):
                # pause
                while cv2.waitKey(1) & 0xFF != ord('p'):
                    pass

            # show result
            out_im = ImageHelpers.imageGrid(ims_out, len(ims_out), 1, 1000)
            cv2.imshow("Frame", out_im)


    def loadVideo(self, vidname):
        """Load a video from file

        Raises:
            FileNotFoundError: Unable to load video with given name

        Args:
            vidname (str): Filename of video containing fish robot tracking markers

        Returns:
            List: Array of frames from the video
        """

        # open video
        vid = cv2.VideoCapture(vidname)

        if not vid.isOpened(): 
            raise FileNotFoundError("Unable to load video with that name, did you make a typo?")

        # read frames to array
        arr = []
        while vid.isOpened():
            ret, im = vid.read()

            if not ret:
                break

            arr.append(im)
        
        print(f"Loaded {len(arr)} frames from video")

        # release and return
        vid.release()
        return arr

    def undistort(self, img):
        """Remove distortion from an image using camera calibration data

        Args:
            img (BGR Image): Image to be undistorted

        Returns:
            BGR Image: Undistorted image
        """
        
        undistorted = cv2.undistort(img, self.calib.mtx, self.calib.dst, None, self.calib.ref_mtx)
        return undistorted