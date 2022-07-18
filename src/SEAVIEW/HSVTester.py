import cv2
import sys
import numpy as np
import imutils

def nothing(x):
    pass

def hsvFilterSelect(frames, init_low=np.array([0,0,0]), init_high=np.array([179,255,255])):
    # Create a window
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
    cv2.createTrackbar('SMin','image',0,255,nothing)
    cv2.createTrackbar('VMin','image',0,255,nothing)
    cv2.createTrackbar('HMax','image',0,179,nothing)
    cv2.createTrackbar('SMax','image',0,255,nothing)
    cv2.createTrackbar('VMax','image',0,255,nothing)

    # Set default value for trackbars.
    cv2.setTrackbarPos('HMax', 'image', init_high[0])
    cv2.setTrackbarPos('SMax', 'image', init_high[1])
    cv2.setTrackbarPos('VMax', 'image', init_high[2])
    cv2.setTrackbarPos('HMin', 'image', init_low[0])
    cv2.setTrackbarPos('SMin', 'image', init_low[1])
    cv2.setTrackbarPos('VMin', 'image', init_low[2])

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    waitTime = 33
    i = 0
    while(1):
        img = imutils.resize(frames[i], height=500)
        cv2.medianBlur(img, 9)
        output = img
   
        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin','image')
        sMin = cv2.getTrackbarPos('SMin','image')
        vMin = cv2.getTrackbarPos('VMin','image')

        hMax = cv2.getTrackbarPos('HMax','image')
        sMax = cv2.getTrackbarPos('SMax','image')
        vMax = cv2.getTrackbarPos('VMax','image')

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(img,img, mask= mask)

        # Print if there is a change in HSV value
        if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            #print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display output image
        cv2.imshow('image', output)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(waitTime) & 0xFF == ord('s'):
            break

        i += 1
        if i >= len(frames):
            i = 0

    cv2.destroyAllWindows()

    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    return lower, upper