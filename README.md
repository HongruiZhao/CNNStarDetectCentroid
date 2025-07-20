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
git clone -b development --single-branch https://github.com/HongruiZhao/CNNStarDetectCentroid.git
cd  CNNStarDetectCentroid
```
Create a conda environment 
```shell
conda create -n CNNStarDetectCentroid python=3.8
conda activate CNNStarDetectCentroid
```
Install pytroch with cuda 11.8
```shell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Install other python packages 
```shell
pip install matplotlib opencv-python scipy pandas tqdm thop gdown
```

## Nightsky videos
Download our nightsky test video `video_Test3.npy` recorded with MT9V022 camera
```shell
cd hardware_experiment
mkdir saved_results
cd saved_results
gdown --id 1iFCgP53cGZ1if_lJWfbFz7qWKfQnFRJI
cd ..
```
* For more non straylight videos, change `id` to `1GxRY8bjWUDtSBRQuXRqOdqgUGGVbaXm_` or `1d-5s3l1tqr7-LotzIVaYLj3xz3nSUDY-`.
* `1MKtcN-BGVzJCeHnqVky1nkWy64VJiUin` for stralight video.
* `1AwpNSWLYxyYel-cjeH1Zpg2WJRxKS88D` for moonlight video. It  does not work very well since the camera was moving around during the recording.


## Data generation 
```shell
cd data_generation
python main_generate_data.py --data 1 --parent_dir "./training_data" --dark_frames_dir "./dark_frames_straylight"
```
this will generate and save 2500 training images, 500 evaluation images, and 500 test images into `training_data` folder using the dark frames from `dark_frames_straylight`.

## Training 
```shell
cd training
python .\training_stepLR.py --trial 1 
```

## Run 
Run with `video_Test3.npy`
```shell
python main_detection_centroiding.py --mode NN --input video --video_file video_Test3.npy
```
* By default it will run our trained model `hardware_experiment/saved_models/MobileUNet_B10_50.pt`.  
* If you want to run ELUnet, go into `hardware_experiment/main_detection_centroiding.py`, function `main_video()`, and comment out MobileUNet and uncomment ELUNet.  
* Changing `--mode` to `baseline` will run the baseline methods defined in function `run_baseline()`.


## Evaluation
Use`hardware_experiment/evaluation.ipynb` to get attitude determination accuracy.