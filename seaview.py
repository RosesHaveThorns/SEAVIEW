from glob import glob
import sys, argparse
from tkinter.tix import Tree
import cv2
import numpy as np

from Calibrate import *

class SeaView():
    
    
    def __init__(self):
        self.calibration = CalibrationData()
        self.CHECKERBOARD_SIZE = (9, 6)

    def calibrate(self):
        global CHECKERBOARD_SIZE

        print("Expecting checkerboard with width={} and height={}".format(self.CHECKERBOARD_SIZE[0], self.CHECKERBOARD_SIZE[1]))

        # initialise calibrator
        calib = Calibrate(self.CHECKERBOARD_SIZE)

        # get images
        calibrating = True
        n_calibimgs = 0
        while calibrating:
            error = None
            img = calib.readImage()
            img_ui = img.copy()

            fnd, corners = calib.isCheckerboard(img)
            if fnd: # draw markers on corners if there is a checkerboard
                img_ui = calib.drawCheckerboardMarkers(img_ui, True, corners)

            # show image
            cv2.putText(img_ui, f"Images: {n_calibimgs}", (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img_ui, f"Calibration Error: {error}", (25,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.putText(img_ui, "-- Controls --", (25,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_ui, "S  add calibration image", (25,85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_ui, "E  update calibration error", (25,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_ui, "Q  quit and save calibration data", (25,115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            
            cv2.imshow("CALIBRATION", img_ui)

            # get key input/wait 1 millis
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and fnd:
                # accuratley find checkerboard in image and analyse
                found = calib.analyseImage(img)
                if found:
                    n_calibimgs += 1
                else:
                    print("Failed to analyse image")

            elif key == ord('e'):
                if n_calibimgs > 0:
                    # calibrate and calculate error. dont save calibration
                    error = round(calib.calculateCalibration(img.shape).error, 3)
                    print("Current error: {error}")
                else:
                    print("No calibration images saved")

            elif key == ord('q'):

                if n_calibimgs > 0:
                    # calibrate and finish
                    self.calibration = calib.calculateCalibration(img.shape)
                    self.calibration.save()
                    print(f"\nCalibration Successful, final error: {round(self.calibration.error, 3)}\n")
                else:
                    print("\nNo calibration images saved, quitting without calibration\n")
                return


    def load_calib(self, filename="camera"):
        self.calibration.load(filename)

    def start(self):
        pass

if __name__ == "__main__":
    # setup command line argument parsing
    parser = argparse.ArgumentParser(description='Track position of a robot fish using a top down camera.')

    parser.add_argument("-c", "--calibrate", action="store_true",
                        help='Force camera calibration during startup')
    parser.add_argument("-C", "--calibfile", default="camera",
                        help='Name of the calibration data file to be used. Do not include ".calib.npz". Ignored if -c flag present')

    args = parser.parse_args()


    # create SEAVIEW instance
    sv = SeaView()

    # use calibration file or calibrate camera
    if args.calibrate:
        sv.calibrate()
    else:
        sv.load_calib(args.calibfile)

    # start SEAVIEW
    sv.start()

    # quit
    cv2.destroyAllWindows()