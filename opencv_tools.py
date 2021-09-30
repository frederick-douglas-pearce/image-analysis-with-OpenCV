#!/usr/bin/env python3
# Code name: opencv_tools.py
# Brief description: Provides a set of image analysis tools mostly derived from OpenCV
# that can be applied to an input image (i.e. numpy array)
#
# Requirements: Python (3.5+?), plus the packages listed below
#
# Start Date: 9/28/21
# Last Revision:
# Current Version:
# Notes:
#
# Copyright (C) 2021, Frederick D. Pearce, opencv_tools.py

# 0. Import modules
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import random

## 1. Define functions
# 1.1 Frame manipulation and display functions
# ToDo: Change these to methods of new class so they can be imported together
def load_frame_gray(img_path, gray_flag=False):
    """Load image at img_path, and convert the original image to grayscale if gray_flag=True.
    Return image and grayscale image if gray_flag=True; otherwise only return original image.
    img_path = a string containing the path to an image file readable by cv.imread
    """
    try:
        img = cv.imread(img_path)
    except Exception as err:
        print(f"The following error occurred when reading the image file at {img_path}: \n{err}")
        img = None            
    if gray_flag and isinstance(img, np.ndarray):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = None
    return (img, gray) if gray_flag else img

def resize_frame(frame, scale=0.5, interp_method = cv.INTER_AREA):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    return cv.resize(frame, (width, height), interpolation=interp_method)

def translate_frame(frame, x, y):
    """Translate an image frame by pixel values (x, y) where
    -x value ---> Left
    -y value ---> Up
    x value ---> Right
    y value ---> Down
    """
    trans_mat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (frame.shape[1], frame.shape[0])
    return cv.warpAffine(frame, trans_mat, dimensions)

def rotate_frame(frame, rotation_angle, rotation_point=None):
    """Rotate an image frame by rotation_angle degrees where
    -rotation_angle value ---> clockwise
    """
    h, w = frame.shape[:2]
    if rotation_point is None:
        rotation_point = (w//2, h//2)
    rotation_matrix = cv.getRotationMatrix2D(rotation_point, rotation_angle, 1.0)
    return cv.warpAffine(frame, rotation_matrix, (w, h))

def flip_frame(frame, flip_code):
    """Flip an image frame using flip_code where
    flip_code = 0 ---> vertical
    flip_code = 1 ---> horizontal
    flip_code = -1 ---> both vertical and horizontal
    """
    return cv.flip(frame, flip_code)

def print_frame_info(frame, frame_desc=""):
    print(f"{frame_desc} Image Shape: Height={frame.shape[0]}, Width={frame.shape[1]}, Channels={frame.shape[2]}")

def show_frame(frame, frame_title):
    cv.imshow(frame_title, frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
# 1.2 Histogram-related functions
def get_hist_params(hist_params, plot_params=None):
    """Return a dictionary containing parameters for calculating and plotting histograms using OpenCV. This
    function defines default parameter values, then updates them based on user input to the function, and finally
    it does error checking to identify incomplete or erroneous input parameters from the user.
    """
    # Default values for all parameters, except 'images', which MUST be provided by user
    params = {
        'hist': {
            'images': None,
            'channels': [0],
            'mask': None,
            'histSize': [256],
            'ranges': [0, 256]
        },
        'plot': {
            'figsize': (10, 8),
            'title': "Image Histogram",
            'xlabel': "Bins",
            'ylabel': "# of Pixels",
            # Specify color code for each channel: MUST have same length as hist 'channels' list above
            'channel_colors': ["k"]
        }
    }
    if 'images' not in hist_params:
        raise KeyError("Missing 'images' key containing a list of images, the only required key/value pair in hist_params")
    # Update param dicts based on user input
    if hist_params:
        try:
            params['hist'].update(hist_params)
        except Exception as e:
            print(e)
    if plot_params:
        try:
            params['plot'].update(plot_params)
        except Exception as e:
            print(e)
    num_channels = len(params['hist']['channels'])
    num_chancols = len(params['plot']['channel_colors'])
    if num_chancols != num_channels:
        raise ValueError(f"# of input channels ({num_channels}) MUST equal # of input channel_colors ({num_chancols})")
    return params

def create_figure_axis(**params):
    plt.figure(figsize=params['figsize'])
    plt.title(params['title'])
    plt.xlabel(params['xlabel'])
    plt.ylabel(params['ylabel'])
    
def calc_plot_histogram(hist, plot):
    for cha, col in zip(hist['channels'], plot['channel_colors']):
        col_hist = cv.calcHist(hist['images'], [cha], hist['mask'], hist['histSize'], hist['ranges'])
        plt.plot(col_hist, color=col)
        plt.xlim(hist['ranges'])

def plot_frame_histogram(hist_params, plot_params=None):
    params = get_hist_params(hist_params, plot_params)
    create_figure_axis(**params['plot'])
    calc_plot_histogram(**params)
    
# 1.3 Edge detection functions
def edges_canny_auto(frame, median_ratio=0.33):
    """Automatic Canny edge detection following https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/"""
    m = np.median(frame)
    l = int(max(0, (1.0-median_ratio)*m))
    h = int(min(255, (1.0-median_ratio)*m))
    return cv.Canny(frame, l, h)

# 1.4 Object detection functions
def detect_all_objects(gray, haar_file, params, object_type="", verbose=False):
    """Return objects detected in input grayscale image, gray, using the haar cascade detector specified in haar_file,
    with the input parameters to the   specified in params.
    """
    # Not the most performant to load haar_cascade for each image when params aren't changing...
    haar_cascade = cv.CascadeClassifier(haar_file)
    detected_objects = haar_cascade.detectMultiScale(gray, **params)
    if verbose:
        print(f"# of {object_type.capitalize()+' ' if object_type else ''}Objects Detected = {len(detected_objects)}")
    return detected_objects

def detect_primary_objects(gray, haar_file, params, num_primary_obj=1, max_iter=10, boost_flag=True, object_type="", verbose=False):
    """Identify the "primary", or least likely to be a false positive, object detected within the input grayscale image, gray.
    The type of object detected by haar_cascade is determined by the haar cascade class xml file that was provided in 
    the haar_file parameter.
    """
    haar_cascade = cv.CascadeClassifier(haar_file)
    detected_objects = haar_cascade.detectMultiScale(gray, **params)
    num_detected = len(detected_objects)
    num_detected_prev = num_detected
    if verbose:
        print(f"\nInitial # of {object_type.capitalize()+' ' if object_type else ''}Objects Detected = {num_detected}")
    num_iter = 0
    boost_factor = 1
    while num_detected != num_primary_obj and num_iter != max_iter:
        num_iter += 1
        if verbose:
            print(f"Iteration # = {num_iter}")
        # Update minNeighbors value in copy of params dict
        if num_iter == 1:
            params_new = params.copy()
        elif num_iter == max_iter:
            print(f"Maximum # of iterations ({max_iter}) reached!")
        # Change minNeighbors up/down depending on whether num detected is too high/low
        # Steps up are twice as big as steps down, and boost_factor determines step size
        if num_detected < num_primary_obj and params_new['minNeighbors'] > 4:
            params_new['minNeighbors'] -= boost_factor
        elif num_detected < num_primary_obj and params_new['minNeighbors'] > 1:
            params_new['minNeighbors'] -= 1            
        elif num_detected > num_primary_obj:
            params_new['minNeighbors'] += 2 * boost_factor
        else:
            print(f"Unable to detect {num_primary_obj} primary object(s) in input image")
            print(f"Verify that either 1) num_detected is zero ({num_detected==0}) and minNeighbors is one ({params_new['minNeighbors']==1})")
            print(f"OR 2) the maximum # of iterations has been reached ({num_iter==max_iter})")
            print("If either of these scenarios occurs, consider changing the input scaleFactor and/or initial minNeighbors value. If neither 1) or 2) applies, then there is an unknown bug somewhere that should be investigated!!!")
        if verbose:
            print(f"minNeighbors = {params_new['minNeighbors']}")
        detected_objects = haar_cascade.detectMultiScale(gray, **params_new)
        num_detected = len(detected_objects)
        if num_detected == num_detected_prev and boost_flag:
            boost_factor += 1
        else:
            boost_factor = 1
        num_detected_prev = num_detected
    if verbose:
        print(f"Final # of {object_type.capitalize()+' ' if object_type else ''}Objects Detected = {num_detected}")
    return detected_objects

def get_detected_objects(gray, detector_params, objects_to_detect=['face'], detector_func=detect_all_objects, verbose=False):
    """High-level function for detecting different types of objects within an input grayscale image, gray."""
    detected_rects = {}
    for object_type in objects_to_detect:
        detected_rects[object_type] = detector_func(gray, object_type=object_type, verbose=verbose, **detector_params[object_type])
    return detected_rects

def get_detected_features_labels(img, detected_rects, label=-1, verbose=False):
    """Loop through each detected rectangle in the input list, detected_rects, and extract the region of interest (ROI)
    from the input image, img, for each detected rectangle. Return a list containing each ROI as the feature, and optionally,
    a list containing the input label value, label, if label > -1.
    """
    obj_rois = []
    for rect in detected_rects:
        try:
            (x, y, w, h) = rect
        except Exception as e:
            print(f"The following error occurred when performing object detection for the image at {img_path}:")
            print(e)
            x = None
        if verbose:
            print(*rect, sep=", ")
        if isinstance(x, int):
            obj_rois.append(img[y:y+h, x:x+w])
        else:
            obj_rois.append(None)
    if label > -1:
        return obj_rois, [label] * len(obj_rois)
    else:
        return obj_rois

def detect_image_objects(gray, detect_params, detect_type="all", label=-1, verbose=False):
    """Detect object(s) in the image located at img_path, using the haar object defined in
    the xml file located at haar_path where
    gray = a grayscale image as an numpy array of type uint8
    detect_params = a dictionary containing two sets of parameters: 1) 'haar_file', a string specifying the full path to the haar cascade
                    xml file to load and 2) 'params' dict to pass to the detectMultiScale method of the haar cascade class. Valid values
                    include scaleFactor (default=1.1), minNeighbors (default=3), and minSize
    detect_type = an optional string specifying the type of detection to perform:
                  "all": runs detect_all_objects, which returns all objects detected from one execution of the haar class detectMultiScale
                  method with the input parameters specified in detect_params. The number of objects detected may vary greatly from image to
                  image for a fixed set of input parameters
                  "primary": runs detect_primary_objects, which performs an iterative process to return a user-specified number of primary objects
                  detected in the input image. Essentially, the minNeighbors parameter is adjusted until the desired number of objects are detected
    label = an optional integer specifying the index to a specific person in the people list that is the primary person in the image at img_path
            When the default value of -1 is provided, then no label is returned (i.e. default is non-training mode)
    verbose = an optional boolean-like value that, when truthy, prints additional details during execution for validation/debugging purposes
    """
    if detect_type == "all":
        detected_rects = detect_all_objects(gray, verbose=verbose, **detect_params)
    elif detect_type == "primary":
        detected_rects = detect_primary_objects(gray, verbose=verbose, **detect_params)
    else:
        print(f"Unrecongized input value for detect_type, {detect_type}, so no objects were detected!")
        print("Please provide a string value for detect_type of either 1) 'all' or 2) 'primary'")
        detected_rects = None
    if isinstance(detected_rects, np.ndarray):
        features_labels = get_detected_features_labels(gray, detected_rects, label=label, verbose=verbose)
        return features_labels
    
def draw_detected_objects(detected_frame, detected_rect, frame_to_show=None, print_detected=False, rect_color=(255, 255, 255), rect_thickness=2):
    """Display source image with detected object(s) outlined, and optionally display an image focused around each detected object based on a
    different input image, frame_to_show. This functionality allow one to show the outline of the detected objects on the grayscale image used
    for detection, and also show the bgr image zoomed in on each detected object.
    detected_frame = numpy array of uint8 specifying the image used for detection
    detected_rect = a list containing zero or more lists, each specifying a rectangle that bounds a detected object in detected_frame
    frame_to_show = an alternate image used to display the image contained within each detected_rect. Input MUST be a numpy array in order
                    to turn this feature on
    print_detected = boolean-like flag when truthy prints the x, y, w, and h values specifying the rectangle bounding each detected object
    rect_color = tuple with three values specifying the (b, g, r) color value for displaying the detected objects
    rect_thickness = integer specifying the thickness of the lines defining the rectangle bounding each detected object
    """
    for i, (x, y, w, h) in enumerate(detected_rect):
        if print_detected:
            print(f"Object {i} Location: x={x}, y={y}, w={w}, h={h}")
        detected_frame = cv.rectangle(detected_frame, (x, y), (x+w, y+h), rect_color, thickness=rect_thickness)
        if isinstance(frame_to_show, np.ndarray):
            show_frame(frame_to_show[y:y+h, x:x+w], "Objects Detected in Image")
    return detected_frame

## 2. If run from command line, execute script below here
if __name__ == "__main__":
    print("ToDo: Implement example that runs as a script!")
