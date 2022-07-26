"""Command line UI for SEAVIEW robotic fish tracking software

Credit: Unless otherwise stated, code by Rose Awen Brindle
"""

# python src/SEAVIEW/seaview.py vids/TEST_nomarkers.mp4 4


import sys, argparse
import cv2
import numpy as np
import imutils

from Calibrate import *
from Tracker import *

class SeaView():
    
    def __init__(self, vid_name, n_markers):
        """Tracking and calibration interface
        """
        self.calibration = CalibrationData()
        self.CHECKERBOARD_SIZE = (9, 6) #(width, height) in squares of calibration checkerboard, defaults to (9, 6)

        self.vid_name = vid_name
        self.n_markers = n_markers

    def track(self):
        """Track robotic fish position in video frames
        """
        if self.calibration.mtx is None:
            print("No calibration data available, stopping...")
            cv2.destroyAllWindows()
            quit()

        tracker = Tracker(self.vid_name, self.n_markers, self.calibration)
        tracker.anaylse()
        tracker.save_centres_csv()
        

    def calibrate(self, subpxl_refinement):
        """Run calibration UI, using images from video file which include checkerboards
        """

        print("Expecting checkerboard with width={} and height={}".format(self.CHECKERBOARD_SIZE[0], self.CHECKERBOARD_SIZE[1]))

        # initialise calibrator
        calib = Calibrate(self.CHECKERBOARD_SIZE, vid=self.vid_name)

        # get images
        calibrating = True
        n_calibimgs = 0
        error = None
        cur_frame = 0
        used_frames = []
        while calibrating:
            img = calib.frames[cur_frame]
            img = imutils.resize(img, width=1000)
            img_ui = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

            col = (0, 0, 255)

            fnd, corners = calib.isCheckerboard(img)
            if fnd: # draw markers on corners if there is a checkerboard
                img_ui = calib.drawCheckerboardMarkers(img_ui, True, corners)
                col = (0, 255, 0)
            

            # show image
            cv2.putText(img_ui, f"Used Frames: {n_calibimgs}", (25,40), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 1, cv2.LINE_AA)
            cv2.putText(img_ui, f"Calibration Error: {error}", (25,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 1, cv2.LINE_AA)
            if cur_frame in used_frames:
                cv2.putText(img_ui, f">>> FRAME USED <<<", (25,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 1, cv2.LINE_AA)

            distfromright = 320
            cv2.putText(img_ui, f"Current Frame: {cur_frame+1}/{len(calib.frames)}", (img_ui.shape[1]-distfromright, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_ui, "               CONTROLS             ", (img_ui.shape[1]-distfromright, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_ui, "S              add calibration image", (img_ui.shape[1]-distfromright, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_ui, "Q     quit and save calibration data", (img_ui.shape[1]-distfromright, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_ui, "[                     previous frame", (img_ui.shape[1]-distfromright, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_ui, "]                         next frame", (img_ui.shape[1]-distfromright, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            cv2.imshow("CALIBRATION", img_ui)

            # get key input/wait 10 millis
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and fnd:
                # accuratley find checkerboard in image and analyse
                found = calib.analyseImage(img)
                if found:
                    n_calibimgs += 1
                    error = round(calib.calculateCalibration(img.shape).error, 3)
                    used_frames.append(cur_frame)
                    print(f"Current error: {error}")
                else:
                    print("Failed to analyse image")
            
            elif key == ord(']'): # use square brackets to move between frames
                cur_frame += 1
                if cur_frame >= len(calib.frames):
                    cur_frame = 0
            elif key == ord('['):
                cur_frame -= 1
                if cur_frame < 0:
                    cur_frame = len(calib.frames)-1

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
                        help='Address of a video including fish tracking markers or calibration checkerboard. Supported formats: MP4')
    parser.add_argument("--num_markers", type=int, default=-1,
                        help='Number of markers to track. If below total markers in image, uses markers closest to the fish\'s head. Required if -c flag is not present')
    

    parser.add_argument("-c", "--calibrate", action="store_true",
                        help='Force camera calibration during startup')
    parser.add_argument("--subpxoff", action="store_true",
                        help='Do not use sub pixel refinement during calibration. Ignored if -c flag is not present')
    parser.add_argument("-C", "--calibfile", default="camera",
                        help='Name of the calibration data file to be used. Do not include ".calib.npz". Ignored if -c flag present')
    parser.add_argument("--cam", default=0, type=int,
                        help='Webcam Id. Ignored if -c flag is not present')

    args = parser.parse_args()

    if args.num_markers < 1 and not args.calibrate:
        print("num_markers argument must be given and above 0 if calibration flag missing")
        parser.print_usage()


    # create SEAVIEW instance
    sv = SeaView(args.videofile, args.num_markers)

    # use calibration file or calibrate camera
    if args.calibrate:
        sv.calibrate(not args.subpxoff)
    else:
        sv.calibration.load(args.calibfile)
        print(f"Loaded Calibration File with error: {round(sv.calibration.error, 3)}")
        sv.track()

    # quit
    cv2.destroyAllWindows()