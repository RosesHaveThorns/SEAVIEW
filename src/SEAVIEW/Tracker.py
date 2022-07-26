"""Contains fish tracking support classes

Credit: Unless otherwise stated, code by Rose Awen Brindle
"""

import cv2
import numpy as np
import csv

from dataclasses import dataclass

import ImageHelpers
import HSVTester

low_blackgreen = np.array([0, 0, 0])
high_blackgreen = np.array([180, 255, 255])

low_black = np.array([0, 0, 0])
high_black = np.array([180, 255, 225])

hsv_filters_filename = "hsvfilters"

@dataclass
class Point:
    x: float
    y: float

class Tracker():

    def __init__(self, vidname, nmarkers, calib):
        global low_blackgreen, high_blackgreen, low_black, high_black

        self.calib = calib

        # load video
        self.frames, self.framenums = ImageHelpers.loadVideo(vidname, scaled_w=2000, every_x=2, max_frames=60, ret_frame_nums=True)
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

        for im, framenum in zip(self.frames, self.framenums):
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
                ROI_PADDING = 20
                for (x, y, r) in final_circles:
                    # calculate top left of marker bounding rect
                    r = r + ROI_PADDING
                    rectX = x - r
                    rectY = y - r

                    # copy ROI of single marker to new empty image
                    mask = np.full((im.shape[0], im.shape[1], 3), np.full((3),230), dtype=np.uint8)
                    mask[rectY:y+r,rectX:x+r] = im[rectY:y+r,rectX:x+r]
                    rois.append(mask)
                    rois_pos.append([rectY, y+r, rectX, x+r])

                    # draw circles and ROI on image
                    cv2.circle(tmp_img, (x, y), r-ROI_PADDING, (0, 0, 255), 4)
                    cv2.rectangle(tmp_img, (rectX, rectY), (x + r, y + r), (0, 128, 255), 2)

                ims_out.append(tmp_img.copy())

                # get centre of marker from cross
                # TODO convex hull then find convexity defects. center point for each defect is middle corners of cross
                # TODO rois are full image, with everything else masked

                tmp_img = im.copy()
                kernel = np.ones((5,5), np.uint8)
                for i in range(len(rois)):

                    # preprocess

                    roi_im = rois[i].copy()
                    #hsv = cv2.cvtColor(roi_im, cv2.COLOR_BGR2HSV)
                    
                    hsv = cv2.GaussianBlur(roi_im, (7, 7), 1.5)

                    #cross_mask = cv2.inRange(hsv, low_black, high_black)
                    #contours_img = cv2.bitwise_and(roi_im, roi_im, mask=cross_mask)

                    #contours_img = cv2.cvtColor(contours_img, cv2.COLOR_HSV2BGR)
                    contours_img_gry = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

                    contours_img = cv2.Canny(hsv, 20, 30)

                    contours_img = cv2.dilate(contours_img, kernel, iterations=1)

                    # copy processed roi to tmp img
                    contours_img_canny = cv2.cvtColor(contours_img, cv2.COLOR_GRAY2BGR)
                    tmp_img[rois_pos[i][0]:rois_pos[i][1], rois_pos[i][2]:rois_pos[i][3]] = contours_img_canny[rois_pos[i][0]:rois_pos[i][1], rois_pos[i][2]:rois_pos[i][3]]

                    # find contours
                    #contours, hierarchy = cv2.findContours(contours_img_gry, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours, hierarchy = cv2.findContours(contours_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

                    # select smallest contour
                    
                    perims = []
                    for i in range(len(contours)):
                        perims.append([cv2.arcLength(contours[i], True), contours[i]])

                    sortedcontours = sorted(perims, key=lambda x: x[0])

                    AREAFILTER = 0.4 # search for smallest contour which is larger than AREAFILTER * next smallest contour
                    i = 0
                    smallestcross = None
                    while True:
                        if sortedcontours[i][0] > sortedcontours[i+1][0] * AREAFILTER:
                            smallestcross = sortedcontours[i][1]
                            break
                        else:
                            i += 1
                    
                    print('Found', len(contours), 'contours in circle')

                    # get convexity defects of contour
                    hull = cv2.convexHull(smallestcross, returnPoints=False)
                    hullfordraw = cv2.convexHull(smallestcross, returnPoints=True)
                    defects = cv2.convexityDefects(smallestcross, hull)


                    # draw contours and hull points
                    cv2.drawContours(tmp_img, contours, -1, (0, 255, 0), 2)
                    cv2.drawContours(tmp_img, [hullfordraw], -1, (255, 0, 0), 2)

                    if defects is not None and len(defects) > 4:
                        print('Found', len(defects), 'convexity defects in cross')
                        pnts = []
                        for i in range(defects.shape[0]):
                            s,e,f,d = defects[i,0]
                            pnt = tuple(smallestcross[f][0])

                            pnts.append([d, pnt])

                        sortedpnts = sorted(pnts, key=lambda x: x[0], reverse=True)

                        # select 4 pnts furthest from hull, ie inner corners
                        innerpnts = []
                        for i in range(4):
                            innerpnts.append(sortedpnts[i][1])

                            cv2.circle(tmp_img, sortedpnts[i][1], 3, [0,0,255], -1)
                        
                        # find centre of 4 points
                        x = int(sum([p[0] for p in innerpnts]) / 4)
                        y = int(sum([p[1] for p in innerpnts]) / 4)
                        centre = Point(x, y)
                        cv2.circle(tmp_img, (centre.x, centre.y), 4, [255,0,0], 2)
                        self.data.append((framenum, centre))
                        

                ims_out.append(tmp_img.copy())
                cv2.imshow("Frame", tmp_img)
                
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
            # out_im = ImageHelpers.imageGrid(ims_out, len(ims_out), 1, 1400)
            # cv2.imshow("Frame", out_im)


    def undistort(self, img):
        """Remove distortion from an image using camera calibration data

        Args:
            img (BGR Image): Image to be undistorted

        Returns:
            BGR Image: Undistorted image
        """
        
        undistorted = cv2.undistort(img, self.calib.mtx, self.calib.dst, None, self.calib.refmtx)
        return undistorted

    def save_centres_csv(self, filename='data.csv'):

        with open(filename, 'w') as f:
            writer = csv.writer(f, dialect='excel')
            writer.writerows([[r[0], r[1].x, r[1].y] for r in self.data])

        print('Saved tracking data as csv')