from sim.star_img_sim import star_img_sim
import numpy as np
from math import radians, sqrt
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import time 
import argparse
import os




def dummy_centroid(gt, pixel_size):
    """
        @param gt: shortest distance transformation graph of centroids. in pixels   
        @param pixel_size: in mm
        @return centroid_est: estimated centroids. a list of [u,v]. in mm
    """

    image_height = gt.shape[0]
    image_width = gt.shape[1] 
    centroid_est = []

    # assign the minimum value within a (3,3) window around each pixel
    gt_min = ndimage.rank_filter(gt, 0, size=(3,3))
      
    for current_row in range(image_height):
        for current_col in range(image_width):
            if (gt[current_row, current_col] == gt_min[current_row, current_col]) and (gt[current_row, current_col] <= (0.5*sqrt(2))):
                u = (current_col + 0.5) * pixel_size
                v = (current_row + 0.5) * pixel_size
                centroid_est.append([u, v])
    
    return centroid_est




def trilateration_centroid(dist_map, seg_map, radius, pixel_size=6/1000.0):
    """
        use points within a 5x5 (at least 5x5) window to determine centroid  
        https://www3.nd.edu/~cpoellab/teaching/cse40815/Chapter10.pdf
        @param dist_map: distance map.
        @param seg_map: segmentation map.   
        @param pixel_size: in mm
        @param radius: radius of the window size, in pixels
        @return centroid_est: estimated centroids. a list of [u,v]. in mm
    """
    image_height = dist_map.shape[0]
    image_width = dist_map.shape[1] 
    centroid_est = []
    threshold = 0.5*sqrt(2) 

    for current_row in range(image_height):
        for current_col in range(image_width):
            if dist_map[current_row, current_col] <= threshold and seg_map[current_row, current_col] > 0  :
                x_i = []
                y_i = []
                r_i = []
                for row_shift in range (-radius, radius+1, 1): 
                    for col_shift in range(-radius, radius+1, 1):
                        row = current_row + row_shift
                        col = current_col + col_shift 
                        if row >= 0 and row < image_height and col >= 0 and col < image_width:
                            if seg_map[row, col] > 0:
                                u = (col + 0.5) 
                                v = (row + 0.5) 
                                x_i.append(u)
                                y_i.append(v)
                                r_i.append(dist_map[row, col])
                                seg_map[row, col] = 0 # avoid getting reused

                n = len(x_i)
                x_n = x_i[-1]
                y_n = y_i[-1]
                r_n = r_i[-1]
                # create matrix A
                A = np.zeros((n-1, 2))
                for i in range(n-1):
                    A[i, 0] = 2 * (x_n - x_i[i])
                    A[i, 1] = 2 * (y_n - y_i[i])

                # create matrix B
                B = np.zeros(n-1)
                for i in range(n-1):
                    B[i] = r_i[i]**2 - r_n**2 - x_i[i]**2 - y_i[i]**2 + x_n**2 + y_n**2

                # caluclate centroid
                A_T = np.transpose(A)
                x = np.linalg.inv(A_T@A) @ A_T @ B
                u = x[0] * pixel_size
                v = x[1] * pixel_size
                centroid_est.append([u, v])

    return centroid_est




def trilateration_centroid_2(gt, radius, pixel_size=6/1000.0):
    """
        use points within a window to determine centroid  
        https://www3.nd.edu/~cpoellab/teaching/cse40815/Chapter10.pdf
        for reciprocal of (segementation mask .* sdt map)
        @param gt: reciprocal of shortest distance transformation, only at star pixels. in pixels.  
        @param pixel_size: in mm
        @param radius: radius of the window size, in pixels
        @return centroid_est: estimated centroids. a list of [u,v]. in mm
    """
    image_height = gt.shape[0]
    image_width = gt.shape[1] 
    centroid_est = []
    threshold = 1 / (0.5*sqrt(2) )

    for current_row in range(image_height):
        for current_col in range(image_width):
            if gt[current_row, current_col] >= threshold:
                x_i = []
                y_i = []
                r_i = []
                for row_shift in range (-radius, radius+1, 1): 
                    for col_shift in range(-radius, radius+1, 1):
                        row = current_row + row_shift
                        col = current_col + col_shift 
                        if row >= 0 and row < image_height and col >= 0 and col < image_width:
                            if gt[row, col] > 0:
                                u = (col + 0.5) 
                                v = (row + 0.5) 
                                x_i.append(u)
                                y_i.append(v)
                                r_i.append(1/gt[row, col])
                                gt[row, col] = 0 # avoid getting reused

                n = len(x_i)
                x_n = x_i[-1]
                y_n = y_i[-1]
                r_n = r_i[-1]
                # create matrix A
                A = np.zeros((n-1, 2))
                for i in range(n-1):
                    A[i, 0] = 2 * (x_n - x_i[i])
                    A[i, 1] = 2 * (y_n - y_i[i])

                # create matrix B
                B = np.zeros(n-1)
                for i in range(n-1):
                    B[i] = r_i[i]**2 - r_n**2 - x_i[i]**2 - y_i[i]**2 + x_n**2 + y_n**2

                # caluclate centroid
                A_T = np.transpose(A)
                x = np.linalg.inv(A_T@A) @ A_T @ B
                u = x[0] * pixel_size
                v = x[1] * pixel_size
                centroid_est.append([u, v])

    return centroid_est




def draw_centroids(centroid, draw_background, pixel_size, color):
    """
        @param centroid: real or estimated centroids.  
        @param draw_background
        @param pixel_size: in mm.  
        @param color: RGB color. for example, green = (0, 1, 0)  
    """

    for i in range(len(centroid)):
        u = centroid[i][0]
        v = centroid[i][1]
        
        # which pixel the star centroid is located
        u_p = int(u // pixel_size)
        v_p = int(v // pixel_size)

        center = (u_p, v_p)
        thickness = 1 # Using thickness of -1 px to fill the circle
        radius = 5
        cv2.circle(draw_background, center, radius, color, thickness)




def gray_to_blue(image):

    image_height = image.shape[0]
    image_width = image.shape[1] 
    image_color = np.zeros((image_height, image_width, 3), dtype=np.float32)
    max_value = np.max(image)

    for current_row in range(image_height):
        for current_col in range(image_width):
            image_color[current_row, current_col, 2] = image[current_row, current_col] / max_value

    return image_color




def compute_centroid_error(centroid_real, centroid_est, pixel_size):
    """
        calculate centroid estimation error.  
        @param centroid_real: star image real centroid coordinates (mm) under the focal plane frame, 2D coordinates + 1 mag + HIP ID.  
        @param centroid_est: estimated centroids. a list of [u,v]. in mm.  
        @param pixel_size: in mm.  
        @return rms: root-mean-square error. in pixels.
        @return true_positive 
        @return false_positive 
        @return false_negative 
    """

    error_list = []
    num_stars_real = len(centroid_real)
    num_stars_est = len(centroid_est)

    for i in range(num_stars_real):
        u_real = centroid_real[i][0]
        v_real = centroid_real[i][1]

        for j in range(len(centroid_est)): 
            u_est = centroid_est[j][0]
            v_est = centroid_est[j][1]
            err = sqrt( (u_real - u_est)**2 + (v_real - v_est)**2 )

            if err < pixel_size:
                error_list.append(err)
                centroid_est.pop(j)
                break
    
    # compute RMS error 
    n = len(error_list)
    rms = 0 
    for item in error_list:
        rms = rms + item*item
    try:
        rms = sqrt(rms / n) / pixel_size
    except ZeroDivisionError:
        pass
    true_positive = n
    false_positive = num_stars_est - true_positive
    false_negative = num_stars_real - true_positive

    return rms, true_positive, false_positive, false_negative 




def visualization(dist_map, seg_map, centroid_real, centroid_est, star_img, pixel_size):
    """
        visualize real centroids and estimated centroids 
    """
    
    # draw centroids
    draw_background_1 = np.zeros((dist_map.shape[0], dist_map.shape[1], 3))
    draw_background_2 = np.zeros((dist_map.shape[0], dist_map.shape[1], 3))
    gt_b = gray_to_blue(seg_map)
    #draw_centroids(centroid_real, draw_background_1, pixel_size, (1,0,0))
    draw_centroids(centroid_est, draw_background_2, pixel_size, (1,0,0))
    draw_result = draw_background_1 + draw_background_2 + gt_b

    plt.figure(1)
    plt.imshow(star_img, cmap='gray', interpolation='none' )

    plt.figure(2)
    plt.imshow(seg_map, cmap='binary', interpolation='none' )
    plt.colorbar()

    plt.figure(3)
    plt.imshow(dist_map, cmap='binary', interpolation='none' )
    plt.colorbar()

    plt.figure(4)
    plt.imshow(draw_result)

    plt.figure(5)
    plt.imshow(np.multiply(seg_map, dist_map) , cmap='binary', interpolation='none' )


    plt.show()




def generate_and_save(star_img_sim_obj, num_sim, split, parent_dir):
    """
        generate and save data for training 
        @param star_img_sim_obj: star image simulator object  
        @param num_sim: number of data generated  
        @param split: "train", "val", or "test"  
        @parent_dir: directory to save data
    """
    if split == "test":
        img_path = os.path.join(parent_dir, "test_raw")
        dist_map_path = os.path.join(parent_dir, "test_dist_map")
        seg_map_path = os.path.join(parent_dir, "test_seg_map")
        centroid_path = os.path.join(parent_dir, "test_centroid")
    elif split == "val":
        img_path = os.path.join(parent_dir, "val_raw")
        dist_map_path = os.path.join(parent_dir, "val_dist_map")
        seg_map_path = os.path.join(parent_dir, "val_seg_map")
        centroid_path = os.path.join(parent_dir, "val_centroid")
    elif split == "train":
        img_path = os.path.join(parent_dir, "train_raw")
        dist_map_path = os.path.join(parent_dir, "train_dist_map")
        seg_map_path = os.path.join(parent_dir, "train_seg_map")
        centroid_path = os.path.join(parent_dir, "train_centroid")
    else:
        print("ERROR: image path not within 'train', 'val' or 'test'")
        return 0

    # create folders
    os.mkdir(img_path)
    os.mkdir(dist_map_path)
    os.mkdir(seg_map_path)
    os.mkdir(centroid_path)

    rng = np.random.default_rng()
    v_input = np.array([0, 0, 0])

    for i in range(num_sim):
        star_time = time.time()
        # random quaterion
        q_input = rng.random(4)
        q_input = q_input/np.linalg.norm(q_input)
        # run star image simulator
        star_img, dist_map, seg_map, centroid_real = star_img_sim_obj.no_smear(q_input, v_input, True)
        # save image 
        np.save(img_path + "/raw_image_{}.npy".format(i), star_img)
        # save distance map
        np.save(dist_map_path + "/dist_map_{}.npy".format(i), dist_map)
        # save segmentation map
        np.save(seg_map_path + "/seg_map_{}.npy".format(i), seg_map)
        # save centroid
        np.save(centroid_path + "/centroid_{}.npy".format(i), centroid_real)
        # print progress
        print("current progress = {} out of {}. time = {:.2f} s".format(i+1, num_sim, time.time()-star_time))

    print("Image Generation Over")
    return 0




if __name__ == "__main__":

    """
        python main_generate_data.py --data 1 --parent_dir "./data_Feb12_2024" --dark_frames_dir "./dark_frames"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=int, default=0)
    parser.add_argument("--parent_dir", type=str, default="./data")
    parser.add_argument("--dark_frames_dir", type=str, default="./dark_frames")

    args = parser.parse_args()

    # create an instant of star image simulatorssss
    star_img_sim_obj = star_img_sim(camera_noise_flag=True, dark_frames_dir=args.dark_frames_dir)
    
    # change some parameters 
    star_img_sim_obj.T_min = 0.1
    star_img_sim_obj.T_max = 0.2
    star_img_sim_obj.sigma_psf_min = 0.5 
    star_img_sim_obj.sigma_psf_max = 1
    star_img_sim_obj.half_width = 5
    # star_img_sim_obj.T = 1 # exposure time
    # star_img_sim_obj.sigma_psf = 0.65
    # star_img_sim_obj.dark_current = 100
    # star_img_sim_obj.read_noise = 100


    if args.data == 1:
        print("Generate training data")
        #generate_and_save(star_img_sim_obj, 1, "train")
        generate_and_save(star_img_sim_obj, 1, "train", args.parent_dir)

        print("Generate validation data")
        #generate_and_save(star_img_sim_obj, 1, "val")
        generate_and_save(star_img_sim_obj, 1, "val", args.parent_dir)

        print("Generate test data")
        #generate_and_save(star_img_sim_obj, 1, "test")
        generate_and_save(star_img_sim_obj, 500, "test", args.parent_dir)

    else:
        print("generate single image, calculate centroids, calculate centroid error, and visualization")
        
        q = np.array([0.707, 0, 0.707, 0]) # least stars 
        # q = np.array([0, -0.707, 0, 0.707])
        # q = np.array([0.5, -0.5, 0.5, 0.5])
        # q = np.array([-0.5, -0.5, -0.5, 0.5])
        # q = np.array([0, 0, 0, 1])
        w = np.array([ radians(0.2), radians(0.5), radians(3) ])
        v_I = np.array([0, 0, 0])
        
        # generate images 
        star_img, dist_map, seg_map, centroid_real = star_img_sim_obj.no_smear(q, v_I, True)
        print(f'num of stars = {len(centroid_real)}')

        # calculate centroids from gt 
        radius = 5
        centroid_est = trilateration_centroid(dist_map.copy(), seg_map.copy(), radius, star_img_sim_obj.pixel_size)
        print(f'num of stars detected = {len(centroid_est)}')

        # calculate centroid error 
        rms, true_positive, false_positive, false_negative  = compute_centroid_error(centroid_real, centroid_est, star_img_sim_obj.pixel_size)
        print(f'root-mean-square error = {rms} pixels \n num of stars correctly detected = {true_positive} \n num of stars incorrectly detected = {false_positive} \n num of stars fail to be detected = {false_negative} ')

        visualization(dist_map, seg_map, centroid_real, centroid_est, star_img, 6/1000)


