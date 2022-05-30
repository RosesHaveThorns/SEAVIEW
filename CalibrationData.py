import cv2
import numpy as np


class CalibrationData(object):

    def __init__(self, mtx, dist, rvecs, tvecs, refined_mtx=None, image_pts=None, object_pts=None):
        """Creates CalibrationData object with initial values"""
        self._mtx = mtx
        self._dist = dist
        self._rvecs = rvecs
        self._tvecs = tvecs
        self._refMtx = refined_mtx
        self._imgpts = image_pts
        self._objpts = object_pts

    def __init__(self):
        """Creates empty CalibrationData objects, for use when loading calibration data from file"""
        self._mtx = None
        self._dist = None
        self._rvecs = None
        self._tvecs = None
        self._refMtx = None
        self._imgpts = None
        self._objpts = None
        
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
        if self._imgpts == None or self._objpts == None:
            return None

        total_error = 0
        for i in range(len(self._imgpts)):
            imgp2, _ = cv2.projectPoints(self._objpts[i], self._rvecs[i], self._tvecs[i], self._mtx, self._dist)
            error = cv2.norm(self._imgpts[i], imgp2, cv2.NORM_L2)/len(imgp2)
            total_error += error

        mean_error = total_error / len(self._objpts)

        return mean_error
    
    def getImagePoints(self):
        """Returns orginal image points array used to create calibration data"""
        return self._imgpts

    def save(self, filename="camera"):
        np.savez_compressed(filename + ".calib", mtx=self._mtx, dist=self._dist, ref_mtx=self._refMtx, rvecs=self._rvecs, tvecs=self._tvecs, imgp=self._imgpts, objp=self._objpts)

    def load(self, filename="camera"):
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
        return "Calibration[mtx:{0}, dist:{1}]".format(self._mtx, self._dist)

    def __str__(self):
        return "Calibration[mtx:{0}, dist:{1}]".format(self._mtx, self._dist)