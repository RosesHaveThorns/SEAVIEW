"""Calibrate: Contains calibration support classes

Classes:
    CalibrationData: A custom datatype for storing camera calibration data
    Calibrate: A class containing helper functions to abstract opencv calibration calculation
"""

from typing import List
import cv2
import numpy as np

class CalibrationData(object):
    
    def __init__(self, mtx=None, dist=None, rvecs=None, tvecs=None, refined_mtx=None, image_pts=None, object_pts=None):
        """Creates CalibrationData object with initial values, all values default to None for use when creating empty object.
        User is expected to run this.load() if any required values are None.

        image_pts and object_pts are not required for image undistortion. All others must be given or loaded before use.
        
        All params are numpy arrays
        """
        self._mtx = mtx
        self._dist = dist
        self._rvecs = rvecs
        self._tvecs = tvecs
        self._refMtx = refined_mtx
        self._imgpts = image_pts
        self._objpts = object_pts
        
    @property
    def mtx(self):
        """Camera calibration matrix"""
        return self._mtx
    
    @property
    def dst(self):
        """__description__"""
        return self._dist
    
    @property
    def rvecs(self):
        """__description__"""
        return self._rvecs

    @property
    def tvecs(self):
        """__description__"""
        return self._tvecs

    @property
    def refmtx(self):
        """Refined camera calibration matrix"""
        return self._refMtx

    @property
    def error(self):
        """Mean error of points in calibration dataset. Returns None if image points or object points are not included in the calib data
        """
        
        if self._imgpts.any() == None or self._objpts.any() == None:
            return None

        total_error = 0
        for i in range(len(self._objpts)):
            imgp2, _ = cv2.projectPoints(self._objpts[i], self._rvecs[i], self._tvecs[i], self._mtx, self._dist)
            error = cv2.norm(self._imgpts[i], imgp2, cv2.NORM_L2)/len(imgp2)
            total_error += error

        mean_error = total_error / len(self._objpts)

        return mean_error
    
    def getImagePoints(self):
        """Returns orginal image points array used to create calibration data"""
        return self._imgpts

    def save(self, filename="camera"):
        """Saves current calibration data to a file with filetype '.calib.npz'

        Args:
            filename (str, optional): Filename of calibration data file, not including '.calib.npz'. Defaults to 'camera'.
        """
        np.savez_compressed(filename + ".calib", mtx=self._mtx, dist=self._dist, ref_mtx=self._refMtx, rvecs=self._rvecs, tvecs=self._tvecs, imgp=self._imgpts, objp=self._objpts)

    def load(self, filename="camera"):
        """Loads a calibration data file with filetype '.calib.npz'

        Args:
            filename (str, optional): Filename of calibration data file, not including '.calib.npz'. Defaults to 'camera'.

        Raises:
            FileNotFoundError: Raised if a calibration file couldn't be found or loaded
        """
        try:
            loaded = np.load(filename + ".calib.npz")
            self._mtx = loaded['mtx']
            self._dist = loaded['dist']
            self._refMtx = loaded['ref_mtx']
            self._rvecs = loaded['rvecs']
            self._tvecs = loaded['tvecs']
            self._imgpts = loaded['imgp']
            self._objpts = loaded['objp']
        except FileNotFoundError:
            raise FileNotFoundError("Could not find calibration file '{0}.calib.npz', are you sure it is in this directory?".format(filename))
        except:
            raise FileNotFoundError("Could not load calibration file, try recalculating and saving camera calibration data.")

    def __repr__(self):
        return "Calibration[mtx:{0}, dist:{1}]".format(self._mtx, self._dist, self.error())

    def __str__(self):
        return "Calibration[mtx:{0}, dist:{1}, error:{2}]".format(self._mtx, self._dist, self.error())

class Calibrate():

    def __init__(self, cb_size, cam_id=0, subpx_refinement = True):
        """Initialise Calibrate object

        Args:
            cb_size (tuple): Size, in squares, of calibration checkerboard in (int, int) format.
            cam_id (int, optional): Webcam id to be used as image source. Defaults to 0.
            subpx_refinement (bool, optional): Use sub pixel refinement during calibration. Defaults to True.
        """

        # initialise calibration data 
        self.calib_data = None

        # initialise empty point location arrays
        self.objectPoints = [] # arrays of locations of points on checkerboard, repeated for each image
        self.imagePoints = [] # arrays of locations of points in each image

        # criteria used by cv2.cornerSubPix to decide when to stop iterating
        self.subpxl_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.subpxl_refine = subpx_refinement

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
        self.calib_data = CalibrationData(mtx, dist, rvecs, tvecs, refined_mtx, np.array(self.imagePoints), np.array(self.objectPoints))
        return self.calib_data


    def analyseImage(self, image):
        """Searches for a checkerboard in the given image and saves corners found if successful. Repeat with multiple images for best accuracy

        Args:
            image (Grayscale Image): A grayscale image in which a chessboard may be present

        Returns:
            bool: Indicates if a chessboard was found in the image
        """

        # find chessboard points in image
        found, corners = cv2.findChessboardCorners(image, self.CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if not found:
            return False

        if self.subpxl_refine:
            # refine point locations to sub pixel accuracy
            corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), self.subpxl_criteria)

        # append image and object points to corresponding arrays for later calibration calculation
        self.imagePoints.append(corners)
        self.objectPoints.append(self.objp)
    
        return True

    def isCheckerboard(self, image):
        """Returns True if a checkerboard in in the image and False if not. Does not effect calibration.

        Args:
            image (Grayscale Image): An image in which a checkerboard may be present

        Returns:
            bool: Indicates if a chessboard was found in the image
            array: list of corners found
        """

        found, corners = cv2.findChessboardCorners(image, self.CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        return found, corners

    def readImage(self):
        """Reads and preprocesses an image from the webcam

        Returns:
            Grayscale Image: opencv grayscale preprocessed image from webcam
        """
        ret, newimg = self.cam.read()

        if not ret:
            print("Failed to read camera")
            image = cv2.imread("error.jpg")

        image = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)

        return image

    def drawCheckerboardMarkers(self, image, found_checkerboard, corners=None):
        """Draw markers showing where checkerboard corners have been recognised on given image. Has no effect on calibration. Use after analyseImage has been called.

        Args:
            image (Grayscale Image): The image to draw markers on
            found_checkerboard (bool): Indicates if a checkerboard is in the image, intended to be passed from isCheckerboard or analyseImage
            corners (array, Optional): Pass the array of checkerboard corners found if using with isCheckerboard. Do not include if using with analyseImage

        Returns:
            BGR Image: Image with markers drawn on
        """

        if corners is None:
            corners = self.imagePoints[-1]
        return cv2.drawChessboardCorners(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), self.CHECKERBOARD, corners, found_checkerboard)