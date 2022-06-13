"""Contains fish tracking support classes

Credit: Unless otherwise stated, code by Rose Awen Brindle
"""

from tkinter import N
import cv2
import numpy as np

import ImageHelpers

low_green = np.array([25, 35, 5])
high_green = np.array([102, 255, 255])

class Tracker():

    def __init__(self, vidname, nmarkers, calib):
        """
        """
        self.calib = calib

        self.frames = self.loadVideo(vidname)
        self.n_markers = nmarkers

        self.data = []

    def anaylse(self):

        for im in self.frames:
            ims_out = [] 

            ims_out.append(im.copy())

            # mask out non-green based on code by robert
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            hsv = cv2.medianBlur(hsv, 9)
            
            blur_ex = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            ims_out.append(blur_ex.copy())

            green_mask = cv2.inRange(hsv, low_green, high_green)
            green_circles = cv2.bitwise_and(im, im, mask=green_mask)

            ims_out.append(green_circles.copy())

            # place mask over white image
            # circle detection works better this way as BLACK cross is not visible (is caught in the mask)
            img_w, img_h, img_c = im.shape
            white = ImageHelpers.makeBlank(img_w, img_h, (255, 255, 255))

            blank_green_circles = cv2.bitwise_and(white, white, mask=green_mask)


            # detect circles in the image
            blank_green_circles = cv2.medianBlur(blank_green_circles, 9)
            circles_flat = blank_green_circles[:,:, 1]
            circles = cv2.HoughCircles(circles_flat, cv2.HOUGH_GRADIENT, 4, 100)

            # check at least num_markers was found
        
            if circles is not None and len(circles[0]) >= self.n_markers-1:
                print(f"Detected {len(circles[0])} in image")

                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")

                tmp = blank_green_circles.copy()

                # select circles with most similar radius
                # order list by r
                # loop through, tracking index of lowest sum of n consecutive numbers

                if len(circles) > self.n_markers-1:
                    circles_sort = np.sort(circles, axis=0)

                    last_max = sum(circles_sort[0:self.n_markers-1])
                    last_max_i = 0 # index for start of most similar numbers

                    for i in range(len(circles_sort)-self.n_markers-1):
                        total = sum(circles_sort[i:i+self.n_markers-1])
                        if total > last_max:
                            last_max = total
                            last_max_i = i

                    final_circles = circles_sort[last_max_i:last_max_i+self.n_markers-1]

                    print(f"Too many circles detected, selecting {self.n_markers-1} with most similar radius:")
                    print(circles_sort)
                    print("Final Selection:")
                    print(final_circles)

                else:
                    final_circles = circles

                # loop over the (x, y) coordinates and radius of the circles
                for (x, y, r) in final_circles:
                    # calculate top left of marker bounding rect
                    rectX = (x - r) 
                    rectY = (y - r)

                    # copy ROI of single marker to new empty image
                    mask = np.zeros(im.shape,np.uint8)
                    mask[rectY:y+r,rectX:x+r] = im[rectY:y+r,rectX:x+r]

                    # draw circles and ROI on image
                    cv2.circle(tmp, (x, y), r, (0, 0, 255), 4)
                    cv2.rectangle(tmp, (rectX, rectY), (x + r, y + r), (0, 128, 255), 2)

                ims_out.append(tmp.copy())

                # TODO convex hull then find convexity defects. center point for each defect is middle corners of cross

            else:
                print("Minimum makers not detected, trying next frame")
                ims_out.append(blank_green_circles.copy())

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