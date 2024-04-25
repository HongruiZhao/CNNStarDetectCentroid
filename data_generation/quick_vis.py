import numpy as np
import argparse
import os 
import glob
import main_generate_data
import matplotlib.pyplot as plt
import time

import sys
sys.path.insert(0, '../star_detection_centroiding/conventional_centroiding')
from threshold_method import centroid_com, detection_globalThreshold, centroiding_CenterOfMass, detection_ST16, centroiding_GaussianGrid, detection_WITM, detection_erosion_dilation


# command: python .\quick_vis.py --folder val --index 0
parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="test")
parser.add_argument("--index", type=float, default=0)
args = parser.parse_args()

img_path = os.path.join( "./data/{}_raw".format(args.folder), "raw_image_{}".format(int(args.index)) )
img_path = glob.glob(img_path + "*.npy")[0]

dist_map_path = os.path.join( "./data/{}_dist_map".format(args.folder), "dist_map_{}".format(int(args.index)) )
dist_map_path = glob.glob( dist_map_path + "*.npy")[0]

seg_map_path = os.path.join( "./data/{}_seg_map".format(args.folder), "seg_map_{}".format(int(args.index)) )
seg_map_path = glob.glob( seg_map_path + "*.npy")[0]

centroid_path = os.path.join( "./data/{}_centroid".format(args.folder), "centroid_{}".format(int(args.index)) )
centroid_path = glob.glob( centroid_path + "*.npy")[0]


star_img = np.load(img_path) 
dist_map = np.load(dist_map_path)
seg_map = np.load(seg_map_path)
centroid_real = np.load(centroid_path)

print(f'num of stars = {len(centroid_real)}')

# # calculate centroids from gt 
# radius = 5
# centroid_est = main_generate_data.trilateration_centroid(dist_map.copy(), seg_map.copy(), radius, 6/1000)
# print(f'num of stars detected = {len(centroid_est)}')

# star detection
start_time = time.time()
#mask = detection_globalThreshold(star_img.copy(), 1.5)
mask = detection_ST16(star_img.copy(), threshold=1.5, pixel_area=6, pixel_sum=50)
#mask = detection_WITM(star_img.copy(), delta=-0.36)
#mask = detection_erosion_dilation( star_img.copy(), gaussian_sigma=2, average_window_size=10, detection_sigma=1.5)

# star centroid
#centroid_est = centroiding_CenterOfMass(star_img.copy(), mask, 1)
centroid_est = centroiding_GaussianGrid(star_img.copy(), mask)
print("--- %s seconds ---" % (time.time() - start_time))
# centroid_est = centroid_com(star_img.copy(), 1.5)

# calculate centroid error 
rms, true_positive, false_positive, false_negative  = main_generate_data.compute_centroid_error(centroid_real, centroid_est.copy(), 6/1000)
print(f'root-mean-square error = {rms} pixels \n num of stars correctly detected = {true_positive} \n num of stars incorrectly detected = {false_positive} \n num of stars fail to be detected = {false_negative} ')

main_generate_data.visualization(dist_map, seg_map, centroid_real, centroid_est, star_img, 6/1000)
#main_generate_data.visualization(dist_map, seg_map+mask, centroid_real, centroid_est, star_img, 6/1000)

