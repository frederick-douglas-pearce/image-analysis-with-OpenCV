Image Analysis with OpenCV in Python
======
* This project contains Jupyter notebooks demonstrating a diverse set of image analysis methods, including python code for scaling, blurring, contour detection, edge detection, masking, visualizing histograms, face detection and face recognition
* The notebooks loosely follow an OpenCV Course made available through freeCodeCamp.org by Jason Dsouza ([available here](https://www.youtube.com/watch?v=oXlwWbU8l2o)). There are currently two notebooks available:
  * opencv_basics_advanced.ipynb: This notebook covers Basic and Advanced image manipulation techniques, including scaling, blurring, contour detection, edge detection, masking, and visualizing histograms. Download example images of the Sun with the link provided to follow along.
  * opencv_faces.ipynb: This notebook focuses on face detection using Haar Cascades and OpenCV's built-in face recognition functionality. Download example images from the Labelled Faces in the Wild (LFW) dataset (via Kaggle) to follow along.

## Usage
1. Clone this repo (or a fork of it)

```
$ git clone https://github.com/frederick-douglas-pearce/image-analysis-with-OpenCV
```

2. Install Requirements
  * **python3** (>3.5 or so) with the following packages installed: jupyterlab, numpy, matplotlib, opencv-contrib-python, caer. You can use pip to install them, but I'd recommend using a virtual environment
  * I used pipenv, a virtual environment and package management tool for python, to install the packages listed above. This repo includes a Pipfile that lists the required packages and any version constraints. The included Pipfiles can also be used to generate a pipenv environment with `pipenv install`

3. Run jupyter lab to open a notebook

```
$ (pipenv run) jupyter lab
```
  * The notebooks contain links to 1) image data and 2) haar cascade model files, which both need to be downloaded in order to run the notebooks


## License
* Copyright 2021 Frederick D. Pearce
* Licensed under the Apache License, Version 2.0 (the "License")
* You may obtain a copy of the License from
[LICENSE](https://github.com/frederick-douglas-pearce/image-analysis-with-OpenCV/blob/main/LICENSE) or
[here](http://www.apache.org/licenses/LICENSE-2.0)
 