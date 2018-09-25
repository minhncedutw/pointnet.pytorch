'''
    File name: record & segment scene
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

#==============================================================================
# Function Definitions
#==============================================================================
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

def generate_ply_from_rgbd(rgb, depth, config):
    points = []
    points_str = []
    for v in range(rgb.shape[1]):
        for u in range(rgb.shape[0]):
            color = rgb[u, v]
            Z = depth[u, v] / config['SCALING_FACTOR']
            if Z == 0:
                continue
            X = (u - config['CENTER_X']) * Z / config['FOCAL_LENGTH']
            Y = (v - config['CENTER_Y']) * Z / config['FOCAL_LENGTH']
            points.append([X, Y, Z, color[0], color[1], color[2], 0])
            points_str.append(f"{X:f} {Y:f} {Z:f} {color[0]:d} {color[1]:d} {color[2]:d} 0\n")
    ply = f"""\
ply
format ascii 1.0
element vertex {len(points_str):d}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
{''.join(points_str)}\
"""
    return ply, points
#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is XXXXXX Program')

    ## Load PointNet config
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model path')
    opt = parser.parse_args()
    print(opt)

    ## Load PointNet model
    num_points = 2700
    classifier = PointNetDenseCls(num_points=num_points, k=10)
    classifier.load_state_dict(torch.load(opt.model))
    classifier.eval()

    ### Config visualization
    cmap = plt.cm.get_cmap("hsv", 5)
    cmap = np.array([cmap(i) for i in range(10)])[:, :3]
    # gt = cmap[seg - 1, :]


    ## Initialize OpenNi
    dist = '/OpenNI-Windows-x64-2.3/Redist'
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

    saving_folder_path = '../DATA/test/'
    if not os.path.exists(saving_folder_path):
        os.makedirs(saving_folder_path+'RGB')
        os.makedirs(saving_folder_path+'D')
        os.makedirs(saving_folder_path+'PC')

    from config import CAMERA_CONFIG

    ## main loop
    s = 1000
    done = False
    while not done:
        key = cv2.waitKey(1) & 255
        ## Read keystrokes
        if key == 27:  # terminate
            print("\tESC key detected!")
            done = True
        elif chr(key) == 's':  # screen capture
            print("\ts key detected. Saving image {}".format(s))
            # cv2.imwrite("ex4_"+str(s)+'.png', canvas)
            # print("/RGB/" + str(s) + '.png')
            rgb = rgb[80:160, 100:200, :]
            dmap = dmap[80:160, 100:200]
            ply_content, points_content = generate_ply_from_rgbd(rgb=rgb, depth=dmap, config=CAMERA_CONFIG)
            cv2.imwrite(saving_folder_path + "RGB/" + str(s) + '.png', rgb)
            cv2.imwrite(saving_folder_path + "D/" + str(s) + '.png', dmap)

            print(rgb.shape, dmap.shape)
            print(type(rgb), type(dmap))
            with open(saving_folder_path + "PC/" + str(s) + '.ply', 'w') as output:
                output.write(ply_content)
            print(saving_folder_path + "PC/" + str(s) + '.ply', ' done')
            s += 1  # uncomment for multiple captures

            ### Get pointcloud of scene for prediction
            points_np = np.array(points_content)[:, :3]
            choice = np.random.choice(len(points_np), num_points, replace=True)
            points_np = points_np[choice, :]
            points_torch = torch.from_numpy(points_np)

            point = point.transpose(1, 0).contiguous()

            point = Variable(point.view(1, point.size()[0], point.size()[1]))

            ### Predict to segment scene
            pred, _ = classifier(point)
            pred_choice = pred.data.max(2)[1]
            print(pred_choice)

        ## Streams
        # RGB
        rgb = get_rgb(rgb_stream=rgb_stream, h=h, w=w)

        # DEPTH
        dmap, d4d = get_depth(depth_stream=depth_stream, h=h, w=w)

        # canvas
        canvas = np.hstack((rgb, d4d))
        ## Display the stream syde-by-side
        cv2.imshow('depth || rgb', canvas)
    # end while

    ## Release resources
    cv2.destroyAllWindows()
    rgb_stream.stop()
    depth_stream.stop()
    openni2.unload()
    print("Terminated")

if __name__ == '__main__':
    main()
