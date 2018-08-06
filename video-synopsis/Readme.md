# Video Analytics: Video Synopsis and Reverse Object Search
This software is based on the following open-source deep learning projects:
1. **[YOLO: Real-Time Object Detection](https://github.com/pjreddie/darknet)** *
2. **[Deep SORT](https://github.com/nwojke/deep_sort)** *

\* **Both the above libraries have their respective "weight" files. Check the description below for downloading and putting the weight files in place.**

### New additions to the software
* We have now provided a __config.ini__ file where a user can provide the path where all the output videos can be written.
* To run the defaults, don't change anything in the config file.
* Path to the output directory should always have a trailing slash.
        __Example: /home/user/Desktop/__


### Video Synopsis
A tool which is used to summarize a large video sequence into a smaller one.

### Reverse Object Search
This is version 1.0 built on string based search. 

### Technology stack
* OpenCV(3.0.0 or higher)
* Python (3.0)
* Tensorflow
* Darknet 

### Environment setup
1. Install virtual environment and virtual environment wrapper
```
$ sudo pip install virtualenv virtualenvwrapper
$ echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.bashrc
$ echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
$ echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
$ cd 
$ . .bashrc
```

2. Setup virtual environment named "video-synopsis"
```
$ mkvirtualenv video-synopsis -p python3 
```
3. Activate virtual environment
```
$ workon video-synopsis
```

4. Install python dependencies inside this virtual environment
```
$ pip install Cython
$ pip install numpy scipy matplotlib ipython pandas sympy nose
$ pip install natsort
$ pip install scikit-learn
$ pip install scikit-image
$ pip install sk-video
$ pip install shapely
$ pip install tensorflow-gpu==1.2.0
```

5. Compile darknet library
**NOTE:** before running this step, download and put weights for the respective libraries at proper places (as per the instructions given below).
```
$ cd video-synopsis/darknet/
$ make -j$(nproc)
```

### Weight files
1. Yolo weights can be downloaded from [here](https://drive.google.com/open?id=1FgZ1MmocnWa7o43Co8B0DHdZYxcJYf92)
2. Deep Sort weights can be downloaded from [here](https://drive.google.com/open?id=1WRBfeJSMd94KS5G05OsUbs_G-T8sNbHv)

### Putting weights
1. Put **yolo.weights** inside `darknet` folder
2. Put **mars-small128.ckpt-68577** & **mars-small128.ckpt-68577.meta** into `deep_sort/resources/networks/` folder

### Instructions
Detailed instructions of how the API works can be found in the [Video-Synopsis-And-Reverse-Object-Search.ipynb](https://github.com/paradigmC/video-analytics/blob/master/video-analytics-api/city/bangalore/video_synopsis/Video-Synopsis-And-Reverse-Object-Search.ipynb) notebook.