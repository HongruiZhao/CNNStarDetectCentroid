<h1 align="center"><strong>Real-Time Convolutional Neural Network-Based Star Detection and Centroiding Method for CubeSat Star Tracker</strong></h1>

<p align="center">
	Hongrui Zhao,
    Michael Lembeck,
    Adrian Zhuang, 
    Riya Shah, 
    Jesse Wei
</p>

<div align="center">
	<a href='https://arxiv.org/abs/2404.19108'><img src='https://img.shields.io/badge/arXiv-2404.19108-b31b1b'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</div>

## Setup
First clone the repository
```shell
git clone https://github.com/HongruiZhao/CNNStarDetectCentroid.git
cd  CNNStarDetectCentroid
```

### 2, Activate conda environment 
* Open "Anaconda Powershell Promopt".  
* Activate conda python environment by ```conda activate ai_star_tracker```.  
* If you named your conda environment differently, you can check all your conda environment names by ```conda info --envs```.   
* You need to install the following packages: 
`pip install matplotlib opencv-python scipy pandas tqdm thop`.


### 3, Run your first AI star tracker demo
* go to the "hardware\star_tracker_simulator_detect" folder by ```cd PATH_TO_FOLDER```.  
* make sure the star videos to be processed are in the folder "saved_results", and the train neural networks are in the folder "saved_models".  You can download test video here: https://drive.google.com/file/d/1iFCgP53cGZ1if_lJWfbFz7qWKfQnFRJI/view?usp=sharing.
* For example, you want to run a demo on "video_Test3.npy" file, run ```python main_detection_centroiding.py --mode NN --input video --video_file video_Test3.npy```.


## Star Tracker Simulator
Go to "star_tracker_test_System\tutorials\README.md", you can details light. 



## Cameras 
### 1, Arducam MT9V022
Go to https://github.com/ArduCAM/ArduCAM_USB_Camera_Shield


