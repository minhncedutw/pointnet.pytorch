'''
    File name: CAMERA_find color range
    Author: minhnc
    Date created(MM/DD/YYYY): 9/22/2018
    Last modified(MM/DD/YYYY HH:MM): 9/22/2018 10:42 AM
    Python Version: 3.5
    Other modules: [tensorflow-gpu 1.3.0]

    Copyright = Copyright (C) 2017 of NGUYEN CONG MINH
    Credits = [None] # people who reported bug fixes, made suggestions, etc. but did not actually write the code
    License = None
    Version = 0.9.0.1
    Maintainer = [None]
    Email = minhnc.edu.tw@gmail.com
    Status = Prototype # "Prototype", "Development", or "Production"
    Code Style: http://web.archive.org/web/20111010053227/ http://jaynes.colorado.edu/PythonGuidelines.html #module_formatting
'''

#==============================================================================
# Imported Modules
#==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import cv2
from primesense import openni2#, nite2
from primesense import _openni2 as c_api

import torch
from torch.autograd import Variable
from pointnet import PointNetDenseCls

#==============================================================================
# Constant Definitions
#==============================================================================
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

#==============================================================================
# Function Definitions
#==============================================================================
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)

def get_rgb(rgb_stream, h, w):
    """
    Returns numpy 3L ndarray to represent the rgb image.
    """
    bgr   = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(), dtype=np.uint8).reshape(h, w, 3)
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def get_depth(depth_stream, h, w):
    """
    Returns numpy ndarrays representing the raw and ranged depth images.
    Outputs:
        dmap:= distancemap in mm, 1L ndarray, dtype=uint16, min=0, max=2**12-1
        d4d := depth for dislay, 3L ndarray, dtype=uint8, min=0, max=255
    Note1:
        fromstring is faster than asarray or frombuffer
    Note2:
        .reshape(120,160) #smaller image for faster response
                OMAP/ARM default video configuration
        .reshape(240,320) # Used to MATCH RGB Image (OMAP/ARM)
                Requires .set_video_mode
    """
    dmap = np.fromstring(depth_stream.read_frame().get_buffer_as_uint16(), dtype=np.uint16).reshape(h, w)  # Works & It's FAST
    d4d = np.uint8(dmap.astype(float) *255/ 2**12-1) # Correct the range. Depth images are 12bits
    d4d = 255 - cv2.cvtColor(d4d,cv2.COLOR_GRAY2RGB)
    return dmap, d4d


#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is XXXXXX Program')

    cv2.namedWindow(window_detection_name)
    cv2.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
    cv2.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
    cv2.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
    cv2.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
    cv2.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
    cv2.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)


    ## Initialize OpenNi
    # dist = './driver/OpenNI-Linux-x64-2.3/Redist'
    dist = './driver/OpenNI-Windows-x64-2.3/Redist'
    openni2.initialize(dist)
    if (openni2.is_initialized()):
        print("openNI2 initialized")
    else:
        print("openNI2 not initialized")

    ## Register the device
    dev = openni2.Device.open_any()

    ## Create the streams stream
    rgb_stream = dev.create_color_stream()
    depth_stream = dev.create_depth_stream()

    ## Define stream parameters
    w = 320
    h = 240
    fps = 30

    ## Configure the rgb_stream -- changes automatically based on bus speed
    rgb_stream.set_video_mode(
        c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=w, resolutionY=h,
                           fps=fps))

    ## Configure the depth_stream -- changes automatically based on bus speed
    # print 'Depth video mode info', depth_stream.get_video_mode() # Checks depth video configuration
    depth_stream.set_video_mode(
        c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=w, resolutionY=h,
                           fps=fps))

    ## Check and configure the mirroring -- default is True
    ## Note: I disable mirroring
    # print 'Mirroring info1', depth_stream.get_mirroring_enabled()
    depth_stream.set_mirroring_enabled(False)
    rgb_stream.set_mirroring_enabled(False)

    ## Start the streams
    rgb_stream.start()
    depth_stream.start()

    ## Synchronize the streams
    dev.set_depth_color_sync_enabled(True)  # synchronize the streams

    ## IMPORTANT: ALIGN DEPTH2RGB (depth wrapped to match rgb stream)
    dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    ## main loop
    done = False
    while not done:
        key = cv2.waitKey(1) & 255
        ## Read keystrokes
        if key == 27:  # terminate
            print("\tESC key detected!")
            done = True

        ## Streams
        # RGB
        rgb = get_rgb(rgb_stream=rgb_stream, h=h, w=w)

        # DEPTH
        dmap, d4d = get_depth(depth_stream=depth_stream, h=h, w=w)

        # canvas
        canvas = np.hstack((rgb, d4d))
        cv2.rectangle(canvas, (119, 79), (202, 162), (0, 255, 0), 1)
        cv2.rectangle(canvas, (119 + 320, 79), (202 + 320, 162), (0, 255, 0), 1)
        ## Display the stream syde-by-side
        cv2.imshow('depth || rgb', canvas)

        hsv = cv2.cvtColor(src=rgb, code=cv2.COLOR_BGR2HSV)

        ### for black
        # tblack = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
        tblack = cv2.inRange(hsv, (100, 130, 0), (130, 220, 150))

        ### for white
        # twhite = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
        twhite = cv2.inRange(hsv, (0, 0, 230, 0), (160, 200, 255, 0))

        cv2.imshow('black', tblack)
        cv2.imshow('white', twhite)
    # end while

    ## Release resources
    cv2.destroyAllWindows()
    rgb_stream.stop()
    depth_stream.stop()
    openni2.unload()
    print("Terminated")

if __name__ == '__main__':
    main()
