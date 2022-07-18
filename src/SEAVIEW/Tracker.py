"""Contains fish tracking support classes

Credit: Unless otherwise stated, code by Rose Awen Brindle
"""

import cv2
import numpy as np

import ImageHelpers
import HSVTester

# - greenblue -
# h 76-161
# s 161-255
# v 0-152

# greenblack
# hMin = 0 , sMin = 36, vMin = 0), (hMax = 170 , sMax = 255, vMax = 210)

# - blue -
# h 95-132
# s 150-255
# v 0-255
# (hMin = 95 , sMin = 147, vMin = 0), (hMax = 132 , sMax = 255, vMax = 255)

# black
# (hMin = 96 , sMin = 69, vMin = 0), (hMax = 179 , sMax = 255, vMax = 224)

# - redblue -
# h 0-179
# s 141-255
# v 88-255

#(hMin = 0 , sMin = 38, vMin = 0), (hMax = 179 , sMax = 255, vMax = 63)
low_blackgreen = np.array([0, 0, 0])
high_blackgreen = np.array([180, 255, 255])

low_black = np.array([0, 0, 0])
high_black = np.array([180, 255, 225])

hsv_filters_filename = "hsvfilters"

class Tracker():

    def __init__(self, vidname, nmarkers, calib):
        global low_blackgreen, high_blackgreen, low_black, high_black

        self.calib = calib

        # load video
        self.frames = ImageHelpers.loadVideo(vidname, scaled_w=1000)
        self.n_markers = nmarkers

        # load last hsv filter values
        try:
            loaded = np.load(hsv_filters_filename + ".hsv.npz")
            low_blackgreen = loaded['low_g']
            high_blackgreen = loaded['high_g']
            low_black = loaded['low_b']
            high_black = loaded['high_b']

        except:
            print(f"Could not load hsv filters file '{hsv_filters_filename}.hsv.npz', defaulting")

        # allow user to change hsv filter vals
        low_blackgreen, high_blackgreen = HSVTester.hsvFilterSelect(self.frames, low_blackgreen, high_blackgreen)
        low_black, high_black = HSVTester.hsvFilterSelect(self.frames, low_black, high_black)

        # save new hsv filter vals
        np.savez_compressed(hsv_filters_filename + ".hsv", low_g=low_blackgreen, high_g=high_blackgreen, low_b=low_black, high_b=high_black)

        self.data = []

    def anaylse(self):
        global low_blackgreen, high_blackgreen, low_black, high_black

        for im in self.frames:
            ims_out = [] 

            #im = self.undistort(im)

            ims_out.append(im.copy())

            # mask out non-green based on code by robert
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            hsv = cv2.GaussianBlur(hsv, (7, 7), 1.5)

            green_mask = cv2.inRange(hsv, low_blackgreen, high_blackgreen)
            green_circles = cv2.bitwise_and(hsv, hsv, mask=green_mask)
            green_circles = cv2.GaussianBlur(green_circles, (7, 7), 2)

            

            # place mask over white image
            # circle detection works better this way as BLACK cross is not visible (is caught in the mask)
            img_w, img_h, img_c = im.shape
            #white = ImageHelpers.makeBlank(img_w, img_h, (255, 255, 255))

            #blank_green_circles = cv2.bitwise_and(white, white, mask=green_mask)


            # detect circles in the image
            #blank_green_circles = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            ims_out.append(green_circles.copy())
            blank_green_circles = cv2.cvtColor(green_circles, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(blank_green_circles, cv2.HOUGH_GRADIENT, 3, 50)

            # check at least num_markers was found
        
            if circles is not None and len(circles[0]) >= self.n_markers-1:
                print(f"Detected {len(circles[0])} circles in image")

                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")

                tmp_img = im.copy()

                # select circles with most similar radius
                # order list by r
                # loop through, tracking index of lowest sum of n consecutive numbers

                if len(circles) > self.n_markers-1:
                    circles_sort = np.sort(circles, axis=0)

                    last_max = np.sum(circles_sort[0:self.n_markers-1])
                    last_max_i = 0 # index for start of most similar numbers

                    for i in range(len(circles_sort)-self.n_markers-1):
                        total = np.sum(circles_sort[i:i+self.n_markers-1])
                        if total > last_max:
                            last_max = total
                            last_max_i = i

                    final_circles = circles_sort[last_max_i:last_max_i+self.n_markers-1]

                    print(f"Too many circles detected, selecting {self.n_markers-1} with most similar radius from:")
                    print(circles_sort)
                    print("Final Selection:")
                    print(final_circles)

                else:
                    final_circles = circles

                # loop over the (x, y) coordinates and radius of the circles
                rois = [] # ROI images
                rois_pos = [] # min and max y and x pos for the roi
                for (x, y, r) in final_circles:
                    # calculate top left of marker bounding rect
                    rectX = (x - r) 
                    rectY = (y - r)

                    # copy ROI of single marker to new empty image
                    mask = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
                    mask[rectY:y+r,rectX:x+r] = im[rectY:y+r,rectX:x+r]
                    rois.append(mask)
                    rois_pos.append([rectY, y+r, rectX, x+r])

                    # draw circles and ROI on image
                    cv2.circle(tmp_img, (x, y), r, (0, 0, 255), 4)
                    cv2.rectangle(tmp_img, (rectX, rectY), (x + r, y + r), (0, 128, 255), 2)

                ims_out.append(tmp_img.copy())

                # get centre of marker from cross
                # TODO convex hull then find convexity defects. center point for each defect is middle corners of cross
                # TODO rois are full image, with everything else masked

                tmp_img = im.copy()
                for i in range(len(rois)):

                    # preprocess

                    # TODO consistent seperation of ONLY cross in each circle
                    # maybe try similar colour cross, but different bright/darkness
                    # or try blue cross, and look for blue and green or blue and red during circle seperation 
                    roi_im = rois[i].copy()
                    #hsv = cv2.cvtColor(roi_im, cv2.COLOR_BGR2HSV)
                    hsv = cv2.GaussianBlur(roi_im, (7, 7), 1.5)

                    #cross_mask = cv2.inRange(hsv, low_black, high_black)
                    #contours_img = cv2.bitwise_and(roi_im, roi_im, mask=cross_mask)

                    #contours_img = cv2.cvtColor(contours_img, cv2.COLOR_HSV2BGR)
                    contours_img_gry = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

                    contours_img = cv2.Canny(hsv, 15, 17)
                    contours_img_canny = cv2.cvtColor(contours_img, cv2.COLOR_GRAY2BGR)

                    # copy processed roi to tmp img
                    tmp_img[rois_pos[i][0]:rois_pos[i][1], rois_pos[i][2]:rois_pos[i][3]] = contours_img_canny[rois_pos[i][0]:rois_pos[i][1], rois_pos[i][2]:rois_pos[i][3]]

                    # find contours
                    contours, hierarchy = cv2.findContours(contours_img_gry, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # get convex hulls of contours
                    hull = []

                    for i in range(len(contours)):
                        hull.append(cv2.convexHull(contours[i], False))

                    # draw contours and hull points
                    #cv2.drawContours(tmp_img, contours, -1, (0, 255, 0), 3, 8, hierarchy)
                    #cv2.drawContours(tmp_img, hull, -1, (255, 0, 0), 3, 8)

                ims_out.append(tmp_img.copy())
                
            else:
                print("Minimum makers not detected, trying next frame")
                ims_out.append(im.copy())
                ims_out.append(im.copy())



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
            out_im = ImageHelpers.imageGrid(ims_out, len(ims_out), 1, 1400)
            cv2.imshow("Frame", out_im)


    def undistort(self, img):
        """Remove distortion from an image using camera calibration data

        Args:
            img (BGR Image): Image to be undistorted

        Returns:
            BGR Image: Undistorted image
        """
        
        undistorted = cv2.undistort(img, self.calib.mtx, self.calib.dst, None, self.calib.refmtx)
        return undistorted