from torch._C import device
from torch.utils import data
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2
import argparse
from conventional_centroiding.threshold_method import centroid_com, detection_globalThreshold, centroiding_CenterOfMass, detection_ST16, centroiding_GaussianGrid, detection_WITM, detection_erosion_dilation
from evaluation import trilateration_centroid, compute_centroid_error
from torchvision.transforms import Normalize




def draw_centroids(centroid, draw_background, pixel_size, color, radius):
    """
        @param centroid: real or estimated centroids.  
        @param draw_background
        @param pixel_size: in mm.  
        @param color: RGB color. for example, green = (0, 1, 0)  
        @param radius: radius of the drawn circle in pixels, should be the same as the window size of the centroid method
        @return draw_background:
    """

    for i in range(len(centroid)):

        u = centroid[i][0]
        v = centroid[i][1]
        
        # which pixel the star centroid is located
        u_p = int(u // pixel_size)
        v_p = int(v // pixel_size)

        center = (u_p, v_p)
        thickness = 1 # Using thickness of -1 px to fill the circle
        cv2.circle(draw_background, center, radius, color, thickness)

    return draw_background




def gray_to_blue(image):

    image_height = image.shape[0]
    image_width = image.shape[1] 
    image_color = np.zeros((image_height, image_width, 3), dtype=np.float32)
    max_value = np.max(image)

    for current_row in range(image_height):
        for current_col in range(image_width):
            image_color[current_row, current_col, 2] = image[current_row, current_col] / max_value

    return image_color




def visualization(error_list, dist_map, seg_map, centroid_real, centroid_est, centroid_est_2, star_img, pixel_size = 6/1000):
    """
        visualize real centroids and estimated centroids 
    """

    # draw centroids
    draw_background_1 = np.zeros((dist_map.shape[0], dist_map.shape[1], 3))
    draw_background_2 = np.zeros((dist_map.shape[0], dist_map.shape[1], 3))
    draw_background_3 = np.zeros((dist_map.shape[0], dist_map.shape[1], 3))

    draw_centroids(centroid_real, draw_background_1, pixel_size, (1,0,0), 8)
    draw_centroids(centroid_est, draw_background_2, pixel_size, (0,1,0), 7)
    draw_centroids(centroid_est_2, draw_background_3, pixel_size, (0,0,1), 6)

    draw_result = draw_background_1 + draw_background_2 + draw_background_3

    plt.figure(1)
    plt.imshow(star_img, cmap='gray', interpolation='none' )
    plt.title("Raw Image")

    plt.figure(2)
    plt.imshow(seg_map, cmap='binary', interpolation='none' )
    plt.colorbar()
    plt.title("Estimated Seg Map")

    plt.figure(3)
    plt.imshow(dist_map, cmap='binary', interpolation='none' )
    plt.colorbar()
    plt.title("Estimated Distance Map")

    plt.figure(4)
    plt.imshow(draw_result)
    plt.title("Detection: Red = True, Green = Est Deep, Blue = Est Conventional")

    plt.figure(5)
    plt.imshow(np.multiply(seg_map, dist_map) , cmap='binary', interpolation='none' )
    plt.title("Esimated Masked Distance Map")

    if error_list != None:
        plt.figure(6)
        # use bins from real stars as bin edges for detected stars 
        real_mag_hist, real_mag_edges = np.histogram(centroid_real[:,2], bins=50)

        error_list = np.asarray(error_list)
        heatmap, xedges, yedges = np.histogram2d(error_list[:,0], error_list[:,1], bins=[40, real_mag_edges])
        heatmap = heatmap / real_mag_hist # each row normalized by real star counts in each magnitude bins. 
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto')
        plt.xlabel("Centroid Error (pixel)")
        plt.ylabel("Star Magnitude")
        plt.title("ML Method: Centroid Error vs Magnitude")

    plt.show()




if __name__ == '__main__':
    from data_load import StarDataSet


    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default="./saved_models/ELUnet_inter_1.pt")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    
    mean = [44.1619381]
    std =  [60.98225565]
    
    pixel_size = 6/1000
    radius = 3
    # load test dataset
    test_dataset = StarDataSet(split='test', norm=False, mean=mean, std=std)
    test_dataloader = data.DataLoader(test_dataset, batch_size=1,
                                        shuffle=True, num_workers=2, 
                                        drop_last=True)                 
    
    # get batch
    batch = next(iter(test_dataloader)) 
    img, dist_map, seg_map, centroid_real  = batch
    centroid_real = torch.squeeze(centroid_real).numpy()
    img = img.to(device)
    
    # get predictions
    model = torch.load(args.load).to(device)
    model.eval()

    prediction = model(Normalize(mean, std)(img)) # only normalize images for neural network so that conventional centroid can use raw images
    seg_prediction = 1.0 * (torch.sigmoid(prediction[0][0]) > 0.5)
    seg_prediction = seg_prediction.detach().cpu().numpy()
    dist_prediction = prediction[0][1]
    dist_prediction = dist_prediction.detach().cpu().numpy()

    img = img[0][0].detach().cpu().numpy()

    # get estimated centroids: machine learning 
    centroid_est = trilateration_centroid(dist_prediction.copy(), seg_prediction.copy(), radius, pixel_size)

    # conventiona detection 
    mask = detection_ST16(img.copy(), threshold=1.5, pixel_area=6, pixel_sum=50)
    #mask = detection_WITM(img.copy(), delta=-0.24)
    #mask = detection_erosion_dilation(img.copy(), gaussian_sigma=2, average_window_size=10, detection_sigma=2.5)


    # conventional centroiding
    #centroid_est_2 = centroiding_CenterOfMass(img.copy(), mask, 5)
    centroid_est_2 = centroiding_GaussianGrid(img.copy(), mask)
    #centroid_est_2 = centroid_com(img.copy(), 5)

    # compute centroid error 
    print(f'num of real centroids: {len(centroid_real)}')
    print(f'num of estimated centroids: {len(centroid_est)}')
    rms, true_positive, false_positive, false_negative, num_stars_real, error_list = compute_centroid_error(centroid_real, centroid_est.copy(), pixel_size)
    print(f'root-mean-square centroid error of the test data set = {rms} pixels. Average true stars {num_stars_real}. Average true positives are {true_positive}. Average false positives are {false_positive}, average false negatives are {false_negative}')

    # compute centroid error for the conventinal method
    print(f'num of real centroids: {len(centroid_real)}')
    print(f'num of estimated centroids: {len(centroid_est_2)}')
    rms, true_positive, false_positive, false_negative, num_stars_real, error_list_2 = compute_centroid_error(centroid_real, centroid_est_2.copy(), pixel_size)
    print(f'root-mean-square centroid error of the test data set = {rms} pixels. Average true stars {num_stars_real}. Average true positives are {true_positive}. Average false positives are {false_positive}, average false negatives are {false_negative}')
   
    # visualization
    centroid_est = np.asarray(centroid_est)
    visualization(error_list, dist_prediction, seg_prediction, centroid_real, centroid_est, centroid_est_2, img, pixel_size)