"""Calibrate: class supporting camera calibration
"""

import cv2
import numpy as np
from CalibrationData import CalibrationData

class Calibrate():

    # criteria used by cv2.cornerSubPix to decide when to stop iterating
    subpxl_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self, cb_size, cam_id=0):
        """Initialise Calibrate object

        Args:
            cb_size (tuple): Size, in squares, of calibration checkerboard in (int, int) format.
            cam_id (int, optional): Webcam id to be used as image source. Defaults to 0.
        """

        # initialise calibration data 
        self.calib_data = None

        # initialise empty point location arrays
        self.objectPoints = [] # arrays of locations of points on checkerboard, repeated for each image
        self.imagePoints = [] # arrays of locations of points in each image

        # convert from size in squares to number of inner corners
        self.CHECKERBOARD = (cb_size[0]-1, cb_size[1]-1)

        # create template object points array
        self.objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

        # get webcam capture object
        self.cam = cv2.VideoCapture(cam_id)

    def getCalibrationData(self):
        """Returns current calibration data object if calibration calculation has been completed

        Raises:
            ValueError: raised if calibration has not been completed yet

        Returns:
            CalibrationData: Current calibration data object
        """

        if self.calib_data == None:
            raise ValueError("Calibration not completed, run calculateCalibration")
        else:
            return self.calib_data
    
    def calculateCalibration(self, image_shape):
        """Calculates camera calibration data from current saved checkerboard points

        Args:
            image_shape (tuple): Height, Width of images used for calibration (must all be same)]

        Return:
            CalibrationData: CalibrationData object containing all required calibration parameters and the image point locations used
        """
        h,w = image_shape[:2]

        # calibration calculations based on checkerboard point locations
        retval, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objectPoints, self.imagePoints, (h,w), None, None)

        # refine camera matrix using initial calibration results
        refined_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # save data to CalibrationData object
        self.calib_data = CalibrationData(mtx, dist, rvecs, tvecs, refined_mtx, self.imagePoints, self.objectPoints)
        return self.calib_data


    def analyseImage(self, image):
        """Searches for a checkerboard in the given image and saves corners found if successful. Repeat with multiple images for best accuracy

        Args:
            image (Grayscale Image): A grayscale image in which a chessboard may be present

        Returns:
            bool: Indicates if a chessboard was found in the image
        """

        global subpxl_criteria

        # find chessboard points in image
        found, corners = cv2.findChessboardCorners(image, self.CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if not found:
            return False

        # refine point locations to sub pixel accuracy
        corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), subpxl_criteria)

        # append image and object points to corresponding arrays for later calibration calculation
        self.imagePoints.append(corners)
        self.objectPoints.append(self.objp)
    
        return True

    def readImage(self):
        """Reads and preprocesses an image from the webcam

        Returns:
            Grayscale Image: opencv grayscale preprocessed image from webcam
        """
        ret, newimg = self.cam.read()
        image = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)

        return image

