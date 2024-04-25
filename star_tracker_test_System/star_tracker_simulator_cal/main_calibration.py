import numpy as np
from camera_software import camera_software
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import time
import pandas as pd
import sys

def get_img(camera_software_obj):
    camera_software_obj.get_images()
    img = np.clip(camera_software_obj.images[0, :, :]*3, a_max=255, a_min=0).astype('uint8') # 0~255 float64 to uint8
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    return img

def main():
    camera_software_obj = camera_software(int_time=5)
    camera_software_obj.open_cameras()

    red_bgr = (0, 0, 255)

    start_time = time.time()
    last_time = start_time
    tmp_list = []
    name = ['time (sec)', 'temperature']
    z = [-7.66069466e-04,  8.05540219e-01, -2.81672337e+02,  3.27728939e+04]
    p = np.poly1d(z)
                
    while True:
        img = get_img(camera_software_obj)
        img = cv2.line(img, (320,0), (320,480), red_bgr, 1)  
        img = cv2.line(img, (0,240), (640,240), red_bgr, 1) 
        cv2.imshow('Calibration', img) 

        tmp_counts = int(camera_software_obj.get_temperature()[0])
        tmp_c = p(tmp_counts)
        print(f'temperature={tmp_c} C, tmp_counts={tmp_counts}')
        # two lines below is to erase the previous print message
        sys.stdout.write("\033[F") # Cursor up one line
        sys.stdout.write("\033[K") # Clear to the end of line ( if you print something shorter than before)

        # if time.time() - last_time > 60: # record every minute
        #     tmp = int(camera_software_obj.get_temperature()[0])
        #     tmp_list.append([time.time()-start_time, tmp])
        #     tmp_csv=pd.DataFrame(columns=name, data=tmp_list)
        #     tmp_csv.to_csv('./tmpcsv.csv',encoding='gbk')
        #     print(f'Time since start = {time.time()-start_time}, temperature={tmp}')
        #     # two lines below is to erase the previous print message
        #     sys.stdout.write("\033[F") # Cursor up one line
        #     sys.stdout.write("\033[K") # Clear to the end of line ( if you print something shorter than before)
        #     last_time = time.time()

        k = cv2.waitKey(1)
        if k == ord('q'):
            print("Close Program")
            break



if __name__ == "__main__":
    main()

