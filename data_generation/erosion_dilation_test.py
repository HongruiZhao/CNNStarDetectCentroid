import numpy as np
import argparse
import os 
import glob
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../star_detection_centroiding/conventional_centroiding')

# libraries for star detection
import cv2 
from scipy import ndimage

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
gaussian_sigma = 2
average_window_size = 10
detection_sigma = 15 # typical value between 10 and 20


# gaussian filter 
# the size of kernel along each axis will be 2*radius + 1
# radius is a new argument for gaussian_filter after scipy 1.10
c = ndimage.gaussian_filter(star_img, sigma=gaussian_sigma, radius=1)

# structural element
b = np.zeros((49,49))
radius = 25
center = (24, 24)
cv2.circle(b, center, radius, 1, -1)

# grey erosion 
cmb = ndimage.grey_erosion(c, structure=b)

# grey dilation
t = ndimage.grey_dilation(cmb, structure=b)

# get background after average filter
B = ndimage.uniform_filter(input=t, size=average_window_size)

# plot structural element 
plt.figure(1)

plt.subplot(2,3,1)
plt.title('Star Image')
plt.imshow(star_img)
plt.colorbar()

plt.subplot(2,3,2)
plt.title('After Gaussian Filter')
plt.imshow(c)
plt.colorbar()

plt.subplot(2,3,3)
plt.title('Structural Element')
plt.imshow(b)
plt.colorbar()

plt.subplot(2,3,4)
plt.title('After Eroision')
plt.imshow(cmb)
plt.colorbar()

plt.subplot(2,3,5)
plt.title('After Dilation')
plt.imshow(t)
plt.colorbar()

plt.subplot(2,3,6)
plt.title('After Average Filter')
plt.imshow(B)
plt.colorbar()

plt.show()








