import tkinter
from PIL import Image, ImageTk
from star_img_sim import star_img_sim
import numpy as np
import time 
import attitude
from math import radians
from tqdm import trange
from itertools import combinations
import cv2 
import matplotlib.pyplot as plt
import pandas as pd

def updateRoot(canvas, image_input):
    image = ImageTk.PhotoImage(Image.fromarray(image_input)) 
    # update the image
    canvas.itemconfig(imgbox, image=image)
    # need to keep a reference of the image, otherwise it will be garbage collected
    canvas.image = image




def visualize_angular_distance(star_img_sim_obj, centroid_mm, star_img_op_sim, selected_stars=None):
    background = np.zeros(( star_img_op_sim.shape[0], star_img_op_sim.shape[1], 3))
    radius = 10 # star circle radius
    red_bgr = (0, 0, 255)
    green_bgr = (0, 255, 0)
    yellow_bgr = (0, 255, 255)
    white_bgr = (255, 255, 255)
    line_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.5
    font_thickness = 1

    pixel_size = star_img_sim_obj.pixel_size
    centroid_mm = np.asarray(centroid_mm)
    if selected_stars is not None:
        centroid_mm = centroid_mm[selected_stars]
    centroid_uv_pixel = (centroid_mm[:,0:2] / pixel_size).astype('int')

    # draw lines and angular distance
    star_vector_array = star_img_sim_obj.star_catalog_array[ centroid_mm[:,4].astype('int'), 0:3]
    dot_product_array = star_vector_array @ np.transpose(star_vector_array)
    angular_distance_array = np.rad2deg(np.arccos(dot_product_array))

    indices = np.arange(angular_distance_array.shape[0])
    comb = np.asarray(list(combinations(indices, 2)))
    angular_distance_centroid = []
    for i in range(comb.shape[0]):
        id1 = comb[i,0]
        id2 = comb[i,1]
        angular_distance = angular_distance_array[id1, id2]
        text = '{:.4f} deg'.format(angular_distance)
        centroid_1 = centroid_uv_pixel[id1]
        centroid_2 = centroid_uv_pixel[id2]
        midpoint = np.abs((centroid_2 + centroid_1)/2).astype('int')
        cv2.line(background, tuple(centroid_1), tuple(centroid_2), green_bgr, line_thickness)
        cv2.putText(background, text, tuple(midpoint), font, fontscale, white_bgr, font_thickness, cv2.LINE_AA)
        
    # draw stars 
    for i in range(centroid_mm.shape[0]):
        cv2.circle(background, tuple(centroid_uv_pixel[i]), radius, red_bgr, -1 )
        text = f'{i}'
        cv2.putText(background, text, tuple(centroid_uv_pixel[i]), font, fontscale*2, yellow_bgr, font_thickness*2, cv2.LINE_AA)


    cv2.imwrite('angular_distance_vis.png', background)

    return angular_distance_array




if __name__ == "__main__":

    #---- INPUT: simulation parameters ----#
    # q: rotation from inertial to star tracker body frame    
    q = np.array([0, 0, 0, 1])
    #q = np.array([0.707, 0, 0.707, 0])  
    #q = np.array([0, -0.707, 0, 0.707])
    #q = np.array([0.5, -0.5, 0.5, 0.5])
    w = np.array([ radians(0), radians(0), radians(0) ]) # angular velocity of rotation from inertial to star tracker body frame
    #w = np.array([ radians(15.042/3600), radians(0), radians(0) ]) # sidereal rate
    #w = np.array([ radians(0), radians(15.042/3600), radians(0) ]) # sidereal rate
    FPS = 28 # capped at 30 FPS
    delay = 1.0/FPS # seconds
    run_time = 60*5 # seconds
    calib_flag = False  
    anime_flag = False
    radius = 30 # radius of the calibration circle
    
    #selected_stars = np.array([0, 1, 2])# indices of stars to visualize angular distance. None to visualize all 
    selected_stars = None 
    #---- INPUT: simulation parameters ----#
    
    
    # create a toplevel widget and enter full screem ,pde 
    root = tkinter.Tk()
    root.attributes('-fullscreen', True) # full screen mode

    # create a canvas and create a placeholder for star images
    canvas = tkinter.Canvas(root, highlightthickness=0)
    canvas.pack(fill=tkinter.BOTH, expand=True) # expand canvas to fill both horizontally and vertically to root
    imgbox = canvas.create_image(0, 0, image=None, anchor='nw')

    # create an instant of star image simulator
    star_img_sim_obj = star_img_sim()


    for _ in trange(0, FPS*run_time, smoothing=0):
        start_time = time.time()

        star_img_op_sim, centroid_mm = star_img_sim_obj.generate_star_img_op_sim(q, calib_flag, radius)
        q = attitude.rk4_kin(delay, w, q, 'quat')

        updateRoot(canvas, star_img_op_sim)
        root.update_idletasks()
        root.update()
        
        if anime_flag and (calib_flag == False):
            while (time.time() - start_time) < delay:
                time.sleep(1e-4) # sleep for a ver short amount of time. time.sleep accuracy depends on OS
        else:
            if calib_flag == False:
                angular_distance_array = visualize_angular_distance(star_img_sim_obj, centroid_mm, star_img_op_sim, selected_stars)
                tmp_csv=pd.DataFrame(data=angular_distance_array)
                tmp_csv.to_csv(f'angular_distance_array.csv',encoding='gbk')
            while True: # sleep forever 
                time.sleep(1)


        

        



