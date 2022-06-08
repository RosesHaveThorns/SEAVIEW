"""Command line UI for SEAVIEW robotic fish tracking software
"""

import sys, argparse
import cv2
import numpy as np

from Calibrate import *
from Tracker import *

class SeaView():
    
    def __init__(self, vid_name, n_markers, camera_id=1):
        """Tracking and calibration interface
        """
        self.calibration = CalibrationData()
        self.CHECKERBOARD_SIZE = (9, 6) #(width, height) in squares of calibration checkerboard, defaults to (9, 6)
        self.cam_id = camera_id

        self.vid_name = vid_name
        self.n_markers = n_markers

    def track(self):
        """Track robotic fish position in video frames
        """
        if self.calibration.mtx is None:
            print("No calibration data available, stopping...")
            cv2.destroyAllWindows()
            quit()

        tracker = Tracker(self.vid_name, self.n_markers)
        

    def calibrate(self, subpxl_refinement):
        """Run calibration UI, using images from webcam which include checkerboards
        """

        print("Expecting checkerboard with width={} and height={}".format(self.CHECKERBOARD_SIZE[0], self.CHECKERBOARD_SIZE[1]))

        # initialise calibrator
        calib = Calibrate(self.CHECKERBOARD_SIZE, cam_id=self.cam_id)

        # get images
        calibrating = True
        n_calibimgs = 0
        error = None
        while calibrating:
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
            cv2.putText(img_ui, "Q  quit and save calibration data", (25,115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            
            cv2.imshow("CALIBRATION", img_ui)

            # get key input/wait 1 millis
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and fnd:
                # accuratley find checkerboard in image and analyse
                found = calib.analyseImage(img)
                if found:
                    n_calibimgs += 1
                    error = round(calib.calculateCalibration(img.shape).error, 3)
                    print(f"Current error: {error}")
                else:
                    print("Failed to analyse image")

            elif key == ord('q'):

                if n_calibimgs > 0:
                    # calibrate and finish
                    self.calibration = calib.calculateCalibration(img.shape)
                    self.calibration.save()
                    print(f"\nCalibration Successful, final error: {round(self.calibration.error, 3)}\n")
                else:
                    print("\nNo calibration images saved, quitting without calibration\n")
                return


if __name__ == "__main__":
    # setup command line argument parsing
    parser = argparse.ArgumentParser(description='Track position of a robot fish using a top down camera.')

    parser.add_argument("videofile", type=str,
                        help='Address of a video including fish tracking markers. Supported formats: MP4')
    parser.add_argument("num_markers", type=int,
                        help='Number of markers to track. If below total markers in image, uses markers closest to the fish\'s head')
    

    parser.add_argument("-c", "--calibrate", action="store_true",
                        help='Force camera calibration during startup')
    parser.add_argument("--subpxoff", action="store_true",
                        help='Do not use sub pixel refinement during calibration. Ignored if -c flag is not present')
    parser.add_argument("-C", "--calibfile", default="camera",
                        help='Name of the calibration data file to be used. Do not include ".calib.npz". Ignored if -c flag present')
    parser.add_argument("--cam", default=0, type=int,
                        help='Webcam Id. Ignored if -c flag is not present')

    args = parser.parse_args()


    # create SEAVIEW instance
    sv = SeaView(args.videofile, args.num_markers, args.cam)

    # use calibration file or calibrate camera
    if args.calibrate:
        sv.calibrate(not args.subpxoff)
    else:
        sv.calibration.load(args.calibfile)
        print(f"Loaded Calibration File with error: {round(sv.calibration.error, 3)}")
        
    sv.track()

    # quit
    cv2.destroyAllWindows()