"""Contains fish tracking support classes
"""

import cv2
import numpy as np

class Tracker():

    def __init__(self, vidname, nmarkers):
        """
        """
        self.frames = self.loadVideo(vidname)
        print(self.frames)

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