# Video Analytics: Slip and fall detection
This software is based on the following open-source library OpenCV

### Slip and fall
An algorithm which can detect an event of slip and fall in case of moving humans.

### Technology stack
* OpenCV(3.0.0 or higher)
* Python (2.7+)

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

3. Activate virtual environment
```
$ workon video-analytics
```
** Assumed that OpenCV has been installed using these [instructions](https://www.learnopencv.com/install-opencv3-on-ubuntu/).

### Instructions
Detailed instructions of how the API works can be found in the `Slip and Fall Detection API.html` notebook (check .html format)

### Sample video link
Sample video can be found [here](https://drive.google.com/open?id=17unXB3F_okAVaieq1epMxvYPeQrdhI24)