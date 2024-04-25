from torch._C import device
from torch.utils import data
import torch
import numpy as np
import argparse
from math import sqrt
from conventional_centroiding.threshold_method import detection_ST16, centroiding_CenterOfMass, detection_globalThreshold, centroiding_GaussianGrid, detection_WITM, detection_erosion_dilation
import sys
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
import time
from scipy.linalg import lstsq



def trilateration_centroid_vectorization(dist_map, seg_map, radius, pixel_size=6/1000.0):
    """
        use points within a 5x5 (at least 5x5) window to determine centroid  
        https://www3.nd.edu/~cpoellab/teaching/cse40815/Chapter10.pdf
        @param dist_map: distance map.
        @param seg_map: segmentation map.   
        @param pixel_size: in mm
        @param radius: radius of the window size, in pixels
        @return centroid_est: estimated centroids. a list of [u,v]. in mm
    """
    dist_map = dist_map.astype('float64')
    seg_map = seg_map.astype('float64')
    image_height = dist_map.shape[0]
    image_width = dist_map.shape[1] 
    centroid_est = []
    threshold = 2 #0.5*sqrt(2) 
    
    coarse_centroid = np.multiply((dist_map <= threshold),  (seg_map > 0))
    row_list, col_list = np.nonzero(coarse_centroid)

    for current_row, current_col in zip(row_list, col_list):

        if seg_map[current_row, current_col] > 0  :

            x_i = []
            y_i = []
            r_i = []
            for row_shift in range (-radius, radius+1, 1): 
                for col_shift in range(-radius, radius+1, 1):
                    row = current_row + row_shift
                    col = current_col + col_shift 
                    try: 
                        if seg_map[row, col] > 0:
                            u = (col + 0.5) 
                            v = (row + 0.5) 
                            r_i.append(dist_map[row, col])
                            x_i.append(u)
                            y_i.append(v)
                            seg_map[row, col] = 0 # avoid getting reused
                    except IndexError:
                        pass
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

            # # caluclate centroid
            # A_T = np.transpose(A)
            # A_TA = A_T@A
            # if np.linalg.cond(A_TA) < 1/sys.float_info.epsilon: # to abandon singular matrix
            #     x = np.linalg.inv(A_TA) @ A_T @ B
            #     u = x[0] * pixel_size
            #     v = x[1] * pixel_size
            #     centroid_est.append([u, v])

            # faster centroid calculation
            # https://stackoverflow.com/questions/55367024/fastest-way-of-solving-linear-least-squares
            x = lstsq(A, B, lapack_driver='gelsy') # sloving Ax=b
            u = x[0][0] * pixel_size
            v = x[0][1] * pixel_size
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
    dist_map = dist_map.astype('float64')
    seg_map = seg_map.astype('float64')
    image_height = dist_map.shape[0]
    image_width = dist_map.shape[1] 
    centroid_est = []
    threshold = 2 #0.5*sqrt(2) 
    
    coarse_centroid = np.multiply((dist_map <= threshold),  (seg_map > 0))
    row_list, col_list = np.nonzero(coarse_centroid)
    
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
                        try: 
                            if seg_map[row, col] > 0:
                                u = (col + 0.5) 
                                v = (row + 0.5) 
                                r_i.append(dist_map[row, col])
                                x_i.append(u)
                                y_i.append(v)
                                seg_map[row, col] = 0 # avoid getting reused
                        except IndexError:
                            pass
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

                # # caluclate centroid
                # A_T = np.transpose(A)
                # A_TA = A_T@A
                # if np.linalg.cond(A_TA) < 1/sys.float_info.epsilon: # to abandon singular matrix
                #     x = np.linalg.inv(A_TA) @ A_T @ B
                #     u = x[0] * pixel_size
                #     v = x[1] * pixel_size
                #     centroid_est.append([u, v])

                # faster centroid calculation
                # https://stackoverflow.com/questions/55367024/fastest-way-of-solving-linear-least-squares
                x = lstsq(A, B, lapack_driver='gelsy') # sloving Ax=b
                u = x[0][0] * pixel_size
                v = x[0][1] * pixel_size
                centroid_est.append([u, v])

    return centroid_est




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
        @return num_stars_real
        @return error_list: [err (pixel), mag]
    """

    error_list = []
    num_stars_real = len(centroid_real)
    num_stars_est = len(centroid_est)

    for i in range(num_stars_real):
        u_real = centroid_real[i][0]
        v_real = centroid_real[i][1]
        mag_real = centroid_real[i][2]

        for j in range(len(centroid_est)): 
            u_est = centroid_est[j][0]
            v_est = centroid_est[j][1]
            err = sqrt( (u_real - u_est)**2 + (v_real - v_est)**2 )

            if err < pixel_size:
                error_list.append([err/pixel_size, mag_real])
                centroid_est.pop(j)
                break
    
    # compute RMS error 
    n = len(error_list)
    rms = 0 
    for err, mag in error_list:
        rms = rms + err*err
    
    try:
        rms = sqrt(rms / n) 
    except ZeroDivisionError:
        pass
    
    true_positive = n
    false_positive = num_stars_est - true_positive
    false_negative = num_stars_real - true_positive

    return rms, true_positive, false_positive, false_negative, num_stars_real, error_list




def evaluation(model, test_dataloader, mean, std, device):
    model.eval()
    model = model.to(device)
    pixel_size = 6/1000.0 # in mm
    radius = 5

    # for ML method 
    rms_total = 0.0 # in pixels
    true_positive_total = 0
    false_positive_total = 0
    false_negative_total = 0
    true_stars_total = 0
    error_mag_list = []

    # for conventional method 
    rms_total_2 = 0.0 # in pixels
    true_positive_total_2 = 0
    false_positive_total_2 = 0
    false_negative_total_2 = 0
    true_stars_total_2 = 0
    error_mag_list_2 = []

    # for saving real star magnitude 
    real_star_magnitude = []

    for counter, batch in enumerate(test_dataloader, 0):
        img, dist_map, seg_map, centroid_real  = batch
        img = img.to(device)
        centroid_real = centroid_real[0].numpy()
        real_star_magnitude.extend(centroid_real[:,2])
        
        # get NN prediction 
        prediction = model(Normalize(mean, std)(img)) # only normalize images for neural network so that conventional centroid can use raw images

        seg_prediction = 1.0 * (torch.sigmoid(prediction[0][0]) > 0.5)
        seg_prediction = seg_prediction.detach().cpu().numpy()
        dist_prediction = prediction[0][1]
        dist_prediction = dist_prediction.detach().cpu().numpy()

        # get estimated centroid
        centroid_est = trilateration_centroid(dist_prediction.copy(), seg_prediction.copy(), radius, pixel_size)

        # conventional star detection
        img = img[0][0].detach().cpu().numpy().copy()
        #mask = detection_ST16(img.copy(), threshold=1.5, pixel_area=6, pixel_sum=50)
        mask = detection_globalThreshold(img.copy(), 1.5)
        #mask = detection_WITM(img.copy(), delta=-0.2)
        #mask = detection_erosion_dilation(img.copy(), gaussian_sigma=2, average_window_size=10, detection_sigma=1.5)

        # conventional centroiding
        centroid_est_2 = centroiding_CenterOfMass(img.copy(), mask, 5)
        #centroid_est_2 = centroiding_GaussianGrid(img.copy(), mask)

        # calculate centroid error 
        rms, true_positive, false_positive, false_negative, num_stars_real, error_list  = compute_centroid_error(centroid_real, centroid_est.copy(), pixel_size)
        rms_total += rms*rms
        true_positive_total += true_positive
        false_positive_total += false_positive
        false_negative_total += false_negative
        true_stars_total += num_stars_real
        error_mag_list.extend(error_list)

        # calculate centroid error for the conventional method 
        rms_2, true_positive_2, false_positive_2, false_negative_2, num_stars_real_2, error_list_2  = compute_centroid_error(centroid_real, centroid_est_2.copy(), pixel_size)
        rms_total_2 += rms_2*rms_2
        true_positive_total_2 += true_positive_2
        false_positive_total_2 += false_positive_2
        false_negative_total_2 += false_negative_2
        true_stars_total_2 += num_stars_real_2
        error_mag_list_2.extend(error_list_2)

        print("evaluation.... {} out of {}".format(counter, len(test_dataloader)))
        torch.cuda.empty_cache()


    rms_total = sqrt(rms_total / len(test_dataloader))
    # false_positive_total /= len(test_dataloader)
    # false_negative_total /= len(test_dataloader)
    # true_positive_total /= len(test_dataloader)
    true_stars_total /= len(test_dataloader)
    precision_1 = true_positive_total / (true_positive_total + false_positive_total)
    recall_1 = true_positive_total / (true_positive_total + false_negative_total)
    F1_1 = 2*precision_1*recall_1 / (precision_1+recall_1)

    rms_total_2 = sqrt(rms_total_2 / len(test_dataloader))
    # false_positive_total_2 /= len(test_dataloader)
    # false_negative_total_2 /= len(test_dataloader)
    # true_positive_total_2 /= len(test_dataloader)
    true_stars_total_2 /= len(test_dataloader)
    precision_2 = true_positive_total_2 / (true_positive_total_2 + false_positive_total_2)
    recall_2 = true_positive_total_2 / (true_positive_total_2 + false_negative_total_2)
    F1_2 = 2*precision_2*recall_2 / (precision_2+recall_2)

    # result_1 = [rms_total, true_positive_total, false_positive_total, false_negative_total, true_stars_total]
    # result_2 = [rms_total_2, true_positive_total_2, false_positive_total_2, false_negative_total_2, true_stars_total_2]
    result_1 = [rms_total, precision_1, recall_1, F1_1, true_stars_total]
    result_2 = [rms_total_2, precision_2, recall_2, F1_2, true_stars_total_2]

    return result_1, result_2, error_mag_list, error_mag_list_2, real_star_magnitude




if __name__ == '__main__':
    from data_load import StarDataSet


    parser = argparse.ArgumentParser()
    #parser.add_argument("--load", type=str, default="./saved_models/ELUnet_B10/ELUnet_1_50.pt")
    #parser.add_argument("--load", type=str, default="./saved_models/MobileUNet_B10/MobileUNet_1_50.pt")
    #parser.add_argument("--load", type=str, default="./saved_models/SqueezeUNet_B10/squeeze_unet_1_50.pt")
    parser.add_argument("--load", type=str, default="./saved_models/UNet_B10/CentroidNet_1_50.pt")
    args = parser.parse_args()

    device = torch.device("cuda:0")

    # # for data_straylight
    # mean = [44.1619381]
    # std =  [60.98225565]

    # for data_Oct19
    mean = [25.36114133]
    std =  [44.31162568]

    # load test dataset
    test_dataset = StarDataSet(split='test', norm=False, mean=mean, std=std)
    test_dataloader = data.DataLoader(test_dataset, batch_size=1,
                                        shuffle=True, num_workers=2, 
                                        drop_last=False)        
                

    # load model         
    model = torch.load(args.load)

    # run evaluation
    result_1, result_2, error_mag_list, error_mag_list_2, real_star_magnitude = evaluation(model, test_dataloader, mean, std, device)
    #print(f'ML method: root-mean-square centroid error of the test data set = {result_1[0]} pixels. Average true stars {result_1[4]}. Average true positives are {result_1[1]}. Average false positives are {result_1[2]}, average false negatives are {result_1[3]}')
    #print(f'Conventional method: root-mean-square centroid error of the test data set = {result_2[0]} pixels. Average true stars {result_2[4]}. Average true positives are {result_2[1]}. Average false positives are {result_2[2]}, average false negatives are {result_2[3]}')
    print(f'ML method: root-mean-square centroid error of the test data set = {result_1[0]} pixels. Average true stars {result_1[4]}. Precision={result_1[1]}. Recall={result_1[2]}. F1={result_1[3]}')
    print(f'Conventional method: root-mean-square centroid error of the test data set = {result_2[0]} pixels. Average true stars {result_2[4]}. Precision={result_2[1]}. Recall={result_2[2]}. F1={result_2[3]}')


    # visualize centroid error vs mag
    plt.figure(1)
    # use bins from real stars as bin edges for detected stars 
    real_mag_hist, real_mag_edges = np.histogram(real_star_magnitude, bins=10)
    error_mag_list = np.asarray(error_mag_list)
    heatmap, xedges, yedges = np.histogram2d(error_mag_list[:,0], error_mag_list[:,1], bins=[30, real_mag_edges])
    heatmap = heatmap / real_mag_hist # each row normalized by real star counts in each magnitude bins. 
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto')
    plt.xlabel("Centroid Error (pixel)")
    plt.ylabel("Star Magnitude")
    plt.title("ML Method: Centroid Error vs Magnitude")
    plt.colorbar()
    plt.show()

