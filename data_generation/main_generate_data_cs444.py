from sim.star_img_sim import star_img_sim
from sim.attitude import q_to_a
import numpy as np
from math import radians, sqrt
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import time 
import argparse
import os





def generate_and_save(star_img_sim_obj, num_sim, split):
    """
        generate and save data for training 
        @param star_img_sim_obj: star image simulator object  
        @param num_sim: number of data generated  
        @param split: "train", "val", or "test"  
    """
    parent_dir  = "./data"
    if split == "test":
        centroid_path = os.path.join(parent_dir, "test_centroid")
    elif split == "val":
        centroid_path = os.path.join(parent_dir, "val_centroid")
    elif split == "train":
        centroid_path = os.path.join(parent_dir, "train_centroid")
    else:
        print("ERROR: image path not within 'train', 'val' or 'test'")
        return 0

    # create folders
    os.mkdir(centroid_path)

    rng = np.random.default_rng()
    v_input = np.array([0, 0, 0])

    for i in range(num_sim):
        star_time = time.time()
        # random quaterion
        q_input = rng.random(4)
        q_input = q_input/np.linalg.norm(q_input)
        A_BI_input = q_to_a(q_input)
        # run star image simulator
        centroid_real = star_img_sim_obj.get_real_centroid(A_BI_input, v_input)
        # save centroid
        np.save(centroid_path + "/centroid_{}.npy".format(i), centroid_real)

        # print progress
        print("current progress = {} out of {}. time = {:.2f} s".format(i+1, num_sim, time.time()-star_time))

    print("Image Generation Over")
    return 0




if __name__ == "__main__":

    # create an instant of star image simulator
    star_img_sim_obj = star_img_sim(camera_noise_flag=False)

    print("Generate training data")
    #generate_and_save(star_img_sim_obj, 1, "train")
    generate_and_save(star_img_sim_obj, 20000, "train")

    print("Generate validation data")
    #generate_and_save(star_img_sim_obj, 1, "val")
    generate_and_save(star_img_sim_obj, 5000, "val")

    print("Generate test data")
    #generate_and_save(star_img_sim_obj, 1, "test")
    generate_and_save(star_img_sim_obj, 5000, "test")


