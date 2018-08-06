# Video Analytics: Real-time Car Analytics
This software is based on the following open-source deep learning projects: **[Darkflow](https://github.com/thtrieu/darkflow)** * which is a TensorFlow version of: **[YOLO: Real-Time Object Detection](https://github.com/pjreddie/darknet)**  and **[Deep SORT](https://github.com/nwojke/deep_sort)** algorithms. 

\* Darkflow project requires **"weight"** files. Check the description below for downloading and putting the weight files in place.

## Car Analytics: brief overview:
The purpose of this project is to add object tracking to yolov2 and achieve real-time multiple object tracking. On top of the tracking layer, an interface is provided to include Counting Cars, Speed Estimation, Direction Detection, Traffic Violation Detection.

### New additions to the software
* We have now provided a __config.ini__ file where a user can provide the path where all the output videos can be written.
* To run the defaults, don't change anything in the config file.
* Path to the output directory should always have a trailing slash.
        __Example: /home/user/Desktop/__


### Technology stack
* OpenCV(3.0.0 or higher)
* Python (3.0)
* Tensorflow

###Note: 
* The software requires a GPU to run effectively. 
* It has been tested on Ubuntu 16.04 with a 4GB 940MX Nvidia Graphic card.
* Recommended GPU memory is 8GB.

### Environment setup
1. Install Nvidia Graphics driver, setup CUDA and cuDNN
Follow: http://deeplearning.lipingyang.org/2017/01/18/install-gpu-tensorflow-ubuntu-16-04/

2. Install virtual environment and virtual environment wrapper
```
$ sudo pip install virtualenv virtualenvwrapper
$ echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.bashrc
$ echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
$ echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
$ cd 
$ . .bashrc
```

3. Setup virtual environment named "video-analytics-2" (for Python 2)
```
$ mkvirtualenv video-analytics-2
```

4. Activate virtual environment
```
$ workon video-analytics-2
```

5. Install python dependencies inside this virtual environment
```
$ pip install Cython
$ pip install numpy scipy matplotlib ipython pandas sympy nose
$ pip install natsort
$ pip install scikit-learn
$ pip install scikit-image
$ pip install sk-video
$ pip install shapely
$ pip install tensorflow-gpu==1.2.0
$ pip install FliterPy
$ pip install ipython[all]
```

6. Make a symbolic link for OpenCV library to this environment, if not already done
```
$ cd .virtualenvs/video-analytics-2/lib/python2.7/site-packages/
$ sudo ln -s /usr/local/lib/python2.7/dist-packages/cv2.so cv2.so

``` 


7. Build the Cython extensions in place.
```
$ cd darkflow/
$ python3 setup.py build_ext --inplace
```

8. An important fix is required because the software uses older versions of tensorflow and Python2
Install numpy as a system level Python 3 dependency:
```
sudo pip3 install numpy
```

Append the line below in .bashrc:
```
export CFLAGS="-I /usr/lib/python3/dist-packages/numpy/core/include $CGLAGS"
```

**NOTE:** Download and put weights for the respective libraries at proper places (as per the instructions given below).

### Download links to weight files
1. Darkflow weights are [here](https://drive.google.com/open?id=1FgZ1MmocnWa7o43Co8B0DHdZYxcJYf92)
2. Deep Sort weights are [here](https://drive.google.com/open?id=1WRBfeJSMd94KS5G05OsUbs_G-T8sNbHv)

### Path to put weight files
1. Put **yolo.weights** inside `realtime_car_analytics/darknet/bin/` folder
2. Put **mars-small128.ckpt-68577** & **mars-small128.ckpt-68577.meta** into `realtime_car_analytics/deep_sort/resources/networks/` folder

### API Instructions
Detailed instructions of how the API works can be found in the [Car-Analytics-API-Doc.ipynb](https://github.com/paradigmC/video-analytics/blob/master/video-analytics-api/city/bangalore/realtime_car_analytics/Car-Analytics-API-Doc.ipynb) notebook.