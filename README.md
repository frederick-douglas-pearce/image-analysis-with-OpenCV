Image Analysis with OpenCV in Python
======
This project contains Jupyter notebooks demonstrating a diverse set of image analysis methods in OpenCV, including python code for scaling, blurring, contour detection, edge detection, masking, visualizing histograms, face detection and face recognition. The notebooks were inspired by the freeCodeCamp.org course entitled "OpenCV & Python" by Jason Dsouza ([video link](https://www.youtube.com/watch?v=oXlwWbU8l2o)). The original code for the course is available at Jason Dsouza's [github profile](https://github.com/jasmcaus/opencv-course).

There are currently two notebooks available:
  1. **opencv_basics_advanced.ipynb**: This notebook covers Basic and Advanced image manipulation techniques, including scaling, blurring, contour detection, edge detection, masking, and visualizing histograms. Additions to the course material include
    - Instructions to download example images of the Sun to use for testing
    - A function, `edges_canny_auto`, that automatically selects input parameters based on the approach outlined [here](https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/)
    - A function, `plot_frame_histogram`, that visualizes the distribution of pixel values within an image, including an option to mask the image prior to histogram computation
  2. **opencv_faces.ipynb**: This notebook focuses on Face Detection using Haar Cascades and OpenCV's built-in face recognition functionality. Additions to the course material include
    - Instructions to download example images from the Labelled Faces in the Wild (LFW) dataset (via Kaggle) for testing
    - I've written a collection of functions that streamline and improve the data preprocessing workflow for training a model to recognize people's faces, including
      - A function, `detect_primary_objects`, that uses Haar Cascades to identify a "primary" set of objects within an image, given a user-specified number of objects that should be detected. This is the desired functionality when collecting data to train a face recognizer, as each detected face will be assigned a label, and only the face of the person of interest should be detected, not other faces in the image, clothing, etc. I developed this function after observing numereous false positives using Haar Cascade Face Detection that were being erroneously labeled and included in the dataset used to train the face recognizer. Using the classic approach to Haar Cascade Face detection (i.e. a single multiscale detection with fixed parameters), close to 9% of all faces in the training dataset are false positives that are incorrectly labeled. Replacing this classic approach with the `detect_primary_objects` function eliminates all false positive faces from the example training set highlighted in the notebook, while only slightly increasing (~12%) the runtime to process the data.
      - A function, `show_person_images`, which displays all the face detection rectangles detected for every image of a given person. This provides a quick way to validate the face detection results
      - Several other functions are included that streamline the data processing workflow, including the high-level function, `create_training_data`, that ultimately produces the features (i.e. region of interest containing a face from the image), the label of the person whose face was detected, and all detected face rectangles for each image. A separate python file, `opencv_tools.py`, contains many of the helper functions used to process the data, including `detect_primary_objects`.

Use these image analysis notebooks as a learning tool, a quick resource for functions that perform common image analysis tasks (e.g. face detection), or modify them into templates to quickly test different feature extraction methods for new, image-based projects.

## Usage
1. Clone this repo (or a fork of it)

```
$ git clone https://github.com/frederick-douglas-pearce/image-analysis-with-OpenCV
```

2. Install Requirements
  * **python3** (>3.5 or so) with the following packages installed: jupyterlab, numpy, matplotlib, opencv-contrib-python, caer. You can use pip to install them directly, but I'd recommend using a virtual environment.
  * I used pipenv, a virtual environment and package management tool for python, to install the packages listed above. This repo includes Pipfiles that list the required packages, version constraints, dependencies, etc. The Pipfiles can be used to generate a pipenv environment with `pipenv install` ([pipenv link](https://pipenv.pypa.io/en/latest/)).

3. Run jupyter lab to open a notebook

```
$ (pipenv run) jupyter lab
```
  * Once a JuypterLab session is running in your browser, find the notebook you want to work on using the File Browser in the left panel, then double click on the notebook to open it.
  * The notebooks contain links to download 1) image files and 2) haar cascade model files, which both need to be available from the notebook environment in order to run.


## License
* Copyright 2021 Frederick D. Pearce
* Licensed under the Apache License, Version 2.0 (the "License")
* You may obtain a copy of the License from
[LICENSE](https://github.com/frederick-douglas-pearce/image-analysis-with-OpenCV/blob/main/LICENSE) or
[here](http://www.apache.org/licenses/LICENSE-2.0)
 