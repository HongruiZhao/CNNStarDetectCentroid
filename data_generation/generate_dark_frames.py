import numpy as np
from sim.camera_software import camera_software
import matplotlib.pyplot as plt
import time 
import os 
import glob 
import argparse

"""
    python .\generate_dark_frames.py --vis 0 --int_time 900
"""
parser = argparse.ArgumentParser()
parser.add_argument("--vis", type=int, default=0)
parser.add_argument("--int_time", type=int, default=100)

args = parser.parse_args()

save_dir = "./dark_frames_Oct19"

# set up the camera (integration time set in "camera_software")
camera_software_obj = camera_software()
camera_software_obj.int_time = args.int_time # integration time in ms. Please go between 50~900ms
camera_software_obj.open_cameras()
camera_software_obj.get_images() # Py_ArduCam_writeSensorReg applies after an image is captured

frames_in_folder = len(glob.glob(save_dir + "/*.npy"))
print(f'num of dark frames currently in folder = {frames_in_folder}')
num_of_imgs = 100 # how many dark frames to be generated 

if args.vis == 1:
    print("only visualization, no generatrion")
    # visualization 
    camera_software_obj.get_images()
    raw = camera_software_obj.images[0, :, :]
    raw = np.flip(raw).astype('uint8')
    print(f'mean = {np.mean(raw)}, standard deviation = {np.std(raw)}')
    plt.figure(1)
    plt.imshow( raw, cmap='gray', interpolation='none')
    plt.colorbar()
    plt.show()

else:
    # generating dark frames 
    for i in range(num_of_imgs):
        star_time = time.time()
        camera_software_obj.get_images()
        raw = camera_software_obj.images[0, :, :].astype('uint8')
        np.save( save_dir + "/frame_{}.npy".format(frames_in_folder+i), raw)

        # print progress
        print("current progress = {} out of {}. time = {:.2f} s".format(i+1, num_of_imgs, time.time()-star_time))

    print("Dark Frame Generation Over")




    
