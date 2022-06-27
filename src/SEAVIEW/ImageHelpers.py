"""Image handling helper functions

Credit: Unless otherwise stated, code by Rose Awen Brindle
"""

import itertools
import numpy as np
import imutils
import cv2

def imageGrid(imgs, w, h, final_w=300):
    """Merges a list of BGR images into a single image in a grid format
    All images must have the same shape. If w*h is more than the amount of images passed, empty space if filled white

    Raises:
        ValueError: Not all images have the same shape

    Args:
        imgs (List): List of BGR Images with the same shape
        w (int): Width of grid to be output
        h (int): Height of grid to be output
        final_w (int, Optional): Returned image will be resized to this width, height will scale with same ratio. Defaults to 300

    Returns:
        BGR Image: Single image made up of passed images in a grid

    Credit: Based off of code by Philipp Gorczak, https://gist.github.com/pgorczak/95230f53d3f140e4939c
    """

    n = w*h
    if len(imgs) > n:
        raise ValueError('Number of images ({}) too large for '
                         'matrix size {}x{}'.format(len(imgs), w, h))

    if any(i.shape != imgs[0].shape for i in imgs[1:]):
        raise ValueError('Not all images have the same shape')

    img_h, img_w, img_c = imgs[0].shape

    imgmatrix = np.zeros((img_h * h,
                          img_w * w,
                          img_c),
                         np.uint8)

    imgmatrix.fill(255)    

    positions = itertools.product(range(w), range(h))
    for (x_i, y_i), img in zip(positions, imgs):
        x = x_i * img_w
        y = y_i * img_h
        imgmatrix[y:y+img_h, x:x+img_w, :] = img

    finalimg = imutils.resize(imgmatrix, width = final_w)

    return finalimg

def makeBlank(w, h, colour=(100, 0, 255)):
    """Returns an image of given shape filled with only given colour
    Automaticly selects image depth (c) based on length of 'colour' argument

    Args:
        w (int): Width of required image
        h (int): Height of required image
        colour ((int, int, int), Optional): Colour image should be filled with. Defaults to white, (255, 255, 255)

    Returns:
        BGR Image: Blank image filled with requested colour
    """

    image = np.zeros([w, h, len(colour)], np.uint8)
    image[:] = colour

    return image

def loadVideo(vidname, scaled_w=-1):
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

    count = 1
    while True:
        ret, im = vid.read()

        if not ret:
            print(f"Failed to read frame, attempt {count}")
            count +=1
            if count > 10:
                break
            continue

        if scaled_w > 0:
            im = imutils.resize(im, width=scaled_w)
        
        count = 1
        arr.append(im)

        if len(arr) % 20 == 0:
            print(f"Loaded frame {len(arr)}")
    
    print(f"Loaded {len(arr)} frames from video")

    # release and return
    vid.release()
    return arr