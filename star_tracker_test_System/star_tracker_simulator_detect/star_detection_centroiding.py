import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
import time 

# cupy
import cupy as cp
import cupyx.scipy.ndimage as ndimagex

def detection_globalThreshold(image_input, factor=5, pixel_area=6):
    """
        flood fill + Liebe adaptive threshold  
        @param image_input: star image as a np array  
        @param factor: 
        @param pixel_area: minimum number of contiguous lit pixels for a valid star detection. 6 in ST16
        @return binary_mask: 1 = star centroid at this pixel, 0 = no star centroid
    """
    
    detection_limit = np.mean(image_input) + factor * np.std(image_input)  # Liebe adaptive threshold
    
    binary_mask = np.zeros(image_input.shape)

    thresh = (image_input > detection_limit)
    thresh.dtype = 'uint8'

    # find contours: curves joining all continous points along the boundary
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # use contours to find coarse centroid
    for cnt in contours:
        M = cv2.moments(cnt)
        num_of_pixels = cv2.contourArea(cnt) 
        # https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html
        # M['m00'] = sum of all pixel values within the contour 
        # M['m10'] = sum of coordinates x (column) weighted by pixel value
        # M['m01'] = sum of coordinates y (row) weighted by pixel value
        if num_of_pixels > pixel_area: 
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            binary_mask[int(cy), int(cx)] = 1

    return binary_mask




def detection_WITM(image_input, delta=-0.5, DELTA=0.2, pixel_area=6):
    """
        weighted iterative threshold method
        https://www.sciencedirect.com/science/article/pii/S0030402613002490  
        @param image_input: star image as a np array  
        @param delta: weight coefficient. in the range of [-0.40, 0]. smaller delta => more star detection, but more noisy
        @param DELTA: threshold for Ti+1 - Ti. The valid range for Δ is [0.20, 0.40]
        @param pixel_area: minimum number of contiguous lit pixels for a valid star detection. 6 in ST16
        @return binary_mask: 1 = star centroid at this pixel, 0 = no star centroid
    """
    image_input = image_input.astype('float64')
    T_old = ( np.max(image_input) + np.min(image_input) ) / 2

    # weighted iterative threshold method
    while True:
        mask = (image_input > T_old)
        not_mask = np.invert(mask)

        R_1 = np.multiply(image_input, mask) # foreground
        R_2 = np.multiply(image_input, not_mask) # background
        n_1 = np.sum(mask) 
        n_2 = np.sum(not_mask)
        mu_1 = np.sum(R_1) / n_1
        mu_2 = np.sum(R_2) / n_2
        T_new = ((1+delta)*mu_1 + (1-delta)*mu_2) / 2
        if abs(T_new - T_old) < DELTA:
            break
        else:
            T_old = T_new

    detection_limit = T_new

    binary_mask = np.zeros(image_input.shape)

    thresh = (image_input > detection_limit)
    thresh.dtype = 'uint8'

    # find contours: curve joining all continous points along the boundary
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # use contours to find coarse centroid
    for cnt in contours:
        M = cv2.moments(cnt)
        num_of_pixels = cv2.contourArea(cnt) 
        # https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html
        # M['m00'] = sum of all pixel values within the contour 
        # M['m10'] = sum of coordinates x (column) weighted by pixel value
        # M['m01'] = sum of coordinates y (row) weighted by pixel value
        if num_of_pixels > pixel_area: 
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            binary_mask[int(cy), int(cx)] = 1

    return binary_mask




def detection_ST16(image_input, threshold=4.3, pixel_area=6):
    """
        ST16 detection routine
        @param image_input: star image as a np array  
        @param threshold: > local average + threshold to be detected. 1.7% of total counts in ST16 
        @param pixel_area: minimum number of contiguous lit pixels for a valid star detection. 6 in ST16
        @return binary_mask: 1 = star centroid at this pixel, 0 = no star centroid
    """
    
    binary_mask = np.zeros(image_input.shape)

    # find local average for each pixel
    image_mean = ndimage.uniform_filter(input=image_input, size=[1,129])

    detection_limit = image_mean + threshold

    thresh = (image_input > detection_limit)
    thresh.dtype = 'uint8'

    # find contours: curve joining all continous points along the boundary
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # use contours to find coarse centroid
    for cnt in contours:
        M = cv2.moments(cnt)
        num_of_pixels = cv2.contourArea(cnt)
        # https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html
        # M['m00'] = sum of all pixel values within the contour 
        # M['m10'] = sum of coordinates x (column) weighted by pixel value
        # M['m01'] = sum of coordinates y (row) weighted by pixel value
        if num_of_pixels > pixel_area: 
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            binary_mask[int(cy), int(cx)] = 1

    return binary_mask




def detection_erosion_dilation(image_input, gaussian_sigma, average_window_size, detection_sigma, pixel_area=6):
    """
        Ref: Motion-blurred star acquisition method of the star tracker under high dynamic conditions
        @param image_input: star image as a np array  
        @param gaussian_sigma: sigma for Gaussian filter. should be consistent with star point spread energy distribution
        @param average_window_size: the size of average filter window 
        @param detection_sigma: threshold = B + sigma. sigma = 10~20 according to the paper.
        @param pixel_area: minimum number of contiguous lit pixels for a valid star detection. 6 in ST16
    """
    
    binary_mask = np.zeros(image_input.shape)

    # gaussian filter 
    # the size of kernel along each axis will be 2*radius + 1
    # radius is a new argument for gaussian_filter after scipy 1.10
    c = ndimage.gaussian_filter(image_input, sigma=gaussian_sigma, radius=1)
    
    """
        structural element
        the radius of structural element b needs to be larger than the radius of the stellar image to capture background 
        but large element takes longer time.
        use cupy to utilize GPU
        
        -install cupy: https://cupy.dev/
        # For CUDA 11.2 ~ 11.x
        pip install cupy-cuda11x
    """
 
    b = np.zeros((49,49))
    radius = 25
    center = (24, 24)
    cv2.circle(b, center, radius, 1, -1)
    b = cp.asarray(b)

    # grey erosion 
    #cmb = ndimage.grey_erosion(c, structure=b)
    c = cp.asarray(c)
    cmb = ndimagex.minimum_filter(c, footprint=b)

    # grey dilation
    #t = ndimage.grey_dilation(cmb, structure=b)
    t = ndimagex.maximum_filter(cmb, footprint=b)
    t = cp.asnumpy(t)

    # get background after average filter
    B = ndimage.uniform_filter(input=t, size=average_window_size)
    detection_limit = B + detection_sigma

    thresh = (image_input > detection_limit)
    thresh.dtype = 'uint8'

    # FOR DEBUG
    # plt.subplot(3,1,1)
    # plt.imshow(thresh)
    # plt.subplot(3,1,2)
    # plt.imshow(detection_limit, cmap='hot')
    # plt.subplot(3,1,3)
    # plt.imshow(image_input, cmap='hot')
    # plt.show()

    # find contours: curve joining all continous points along the boundary
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # use contours to find coarse centroid
    for cnt in contours:
        M = cv2.moments(cnt)
        num_of_pixels = cv2.contourArea(cnt)
        # https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html
        # M['m00'] = sum of all pixel values within the contour 
        # M['m10'] = sum of coordinates x (column) weighted by pixel value
        # M['m01'] = sum of coordinates y (row) weighted by pixel value
        if num_of_pixels > pixel_area: 
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            binary_mask[int(cy), int(cx)] = 1

    return binary_mask





def centroiding_CenterOfMass(image_input, mask_input, window_size):
    """
        center of mass centroiding 
        @param image_input: star image as a np array          
        @param mask_input: = 1 if star centroid is here, = 0 otherwise.
        @param window_size: (window_size*2+1) x (window_size*2+1) window around the brightest pixel to compute centroid 
        @return centroid_result: [u, v, intensity]
    """

    pixel_size = 1 # =1 so unit is pixel instead of mm
    centroid_result = [] # a list whose row element is [ centroid_u, centroid_v, sum_I ]

    row_list, col_list = np.nonzero(mask_input)
    for current_row, current_col in zip(row_list, col_list):

        ### centroid computation ###

        # Ref: A software package for evaluating the performance of a star sensor operation
        u_sum_xI = 0
        v_sum_yI = 0
        sum_I = 0
        for col_bri_neigh in range(current_col-window_size, current_col+window_size+1, 1):
            for row_bri_neigh in range(current_row-window_size, current_row+window_size+1, 1):
                try:
                    u_sum_xI = u_sum_xI + pixel_size * (col_bri_neigh + 0.5) * image_input[row_bri_neigh, col_bri_neigh] # pixel coordinate starts from 0, pick the center of a pixel 
                    v_sum_yI = v_sum_yI + pixel_size * (row_bri_neigh + 0.5) * image_input[row_bri_neigh, col_bri_neigh]
                    sum_I = sum_I + image_input[row_bri_neigh, col_bri_neigh]
                except IndexError:
                    pass

        centroid_u = u_sum_xI / sum_I
        centroid_v = v_sum_yI / sum_I
        centroid_result.append([ centroid_u, centroid_v, sum_I ]) # save centroid coordinate and magnitude
        if len(centroid_result) > 50: # too many detection
            return centroid_result    

    return centroid_result




def centroiding_GaussianGrid(image_input, mask_input):
    """
        Gassuain Grid Centroiding Algorithm
        Ref: An Accurate and Efficient Gaussian Fit Centroiding Algorithm for Star Trackers 
        @param image_input: star image as a np array          
        @param mask_input: = 1 if star centroid is here, = 0 otherwise.
        @return centroid_result: [u, v, intensity]
    """

    pixel_size = 1 # =1 so unit is pixel instead of mm
    centroid_result = [] # a list whose row element is [ centroid_u, centroid_v, sum_I ]
    background_value = 10 # mean value of a dark frame to fill pixel value out of bound
    
    row_list, col_list = np.nonzero(mask_input)
    for current_row, current_col in zip(row_list, col_list):
                
        ### centroid computation start###
        V = np.zeros((5,5)) # matrix to hold pixel value within a 5x5 window 
        for row in range(5):
            for col in range(5):
                try:
                    V[row, col] = image_input[current_row+(row-2), current_col+(col-2)]
                except IndexError:
                    V[row, col] = background_value
        # coordinate of the center of the window             
        xc = pixel_size * current_col 
        yc = pixel_size * current_row
        # get xb 
        Aycm2, Bycm2 = GaussainGrid_AB(V[0,:])
        Aycm1, Bycm1 = GaussainGrid_AB(V[1,:])
        Ayc, Byc = GaussainGrid_AB(V[2,:])
        Ayc1, Byc1 = GaussainGrid_AB(V[3,:])
        Ayc2, Byc2 = GaussainGrid_AB(V[4,:])
        A = np.hstack((Aycm2, Aycm1, Ayc, Ayc1, Ayc2))
        B = np.hstack((Bycm2, Bycm1, Byc, Byc1, Byc2))
        Vx = np.log( np.hstack((V[0,:], V[1,:], V[2,:], V[3,:], V[4,:])) )  
        xb = xc + pixel_size*np.dot(B,Vx)/np.dot(A,Vx)
        # get yb
        Axcm2, Bxcm2 = GaussainGrid_AB(V[:,0])
        Axcm1, Bxcm1 = GaussainGrid_AB(V[:,1])
        Axc, Bxc = GaussainGrid_AB(V[:,2])
        Axc1, Bxc1 = GaussainGrid_AB(V[:,3])
        Axc2, Bxc2 = GaussainGrid_AB(V[:,4])
        A = np.hstack((Axcm2, Axcm1, Axc, Axc1, Axc2))
        B = np.hstack((Bxcm2, Bxcm1, Bxc, Bxc1, Bxc2))
        Vy = np.log( np.hstack((V[:,0], V[:,1], V[:,2], V[:,3], V[:,4])) )  
        yb = yc + pixel_size*np.dot(B,Vy)/np.dot(A,Vy)

        centroid_result.append([ xb, yb, np.sum(V) ]) # save centroid coordinate and magnitude
        if len(centroid_result) > 50: # too many detection
            return centroid_result
        ### centroid computation end ###


    return centroid_result




def GaussainGrid_AB(W):
    """
        Calculate A and B vector for Gaussian Grid Centroiding method 
        @param W: a list of weight W = [w-2, w-1, w0, w1, w2]
        @return A: A = [A-2, A-1, A0, A1, A2]
        @return B: B = [B-2, B-1, B0, B1, B2]
    """

    wm2 = W[0]
    wm1 = W[1]
    w0 = W[2]
    w1 = W[3]
    w2 = W[4]
    
    A0 = 2*(w0*w1*w2 + w0*wm1*wm2) - 4*w0*w1*wm1 - 18*(w0*w1*wm2 + w0*wm1*w2) - 64*w0*w2*wm2
    A1 = 2*w0*wm1*w1 + 6*w1*wm1*wm2 - 4*w0*w1*w2 + 12*w0*w1*wm2 - 18*w1*w2*wm1 - 48*w1*w2*wm2
    A2 = 2*w0*w1*w2 + 6*w0*w2*wm1 + 12*(w1*w2*wm1 + w2*wm1*wm2) + 32*w0*w2*wm2 + 36*w1*w2*wm2 
    Am1 = 2*w0*w1*wm1 + 6*w1*w2*wm1 - 4*w0*wm1*wm2 + 12*w0*w2*wm1 - 18*w1*wm1*wm2 - 48*w2*wm1*wm2
    Am2 = 2*w0*wm1*wm2 + 6*w0*wm2*w1 + 12*(w1*w2*wm2 + w1*wm1*wm2) + 32*w0*w2*wm2 +36*w2*wm1*wm2
    
    B0 = 3*(w0*w1*w2 - w0*wm1*wm2) + 9*(w0*w1*wm2 - w0*wm1*w2)
    B1 = -w0*w1*wm1 - 4*w0*w1*w2 - 9*(w1*w2*wm1 + w1*wm2*wm1) - 12*w0*w1*wm2
    B2 = w0*w1*w2 - 3*w0*wm1*w2 - 18*(w1*wm2*w2 + wm1*wm2*w2) - 32*w0*w2*wm2                   
    Bm1 = w0*w1*wm1 + 4*w0*wm1*wm2 + 9*(w1*w2*wm1 + w1*wm1*wm2) + 12*w0*w2*wm1
    Bm2 = -w0*wm1*wm2 + 3*w0*w1*wm2 + 18*(w1*w2*wm2 + w2*wm1*wm2) + 32*w0*w2*wm2

    A = np.array([Am2, Am1, A0, A1, A2])
    B = np.array([Bm2, Bm1, B0, B1, B2])

    return A, B






if __name__ == '__main__':
    sky_cam_capture = cv2.VideoCapture('staroutput6.mp4')
    _, sky_cam_frame = sky_cam_capture.read()
    img = cv2.cvtColor(sky_cam_frame, cv2.COLOR_BGR2GRAY)       

    #detection_globalThreshold(img.copy(), factor=5)   
    #detection_WITM(img.copy(), delta=-0.24, DELTA=0.3)
    #detection_ST16(img.copy(), threshold=50 )
    detection_erosion_dilation(img.copy(), gaussian_sigma=2, average_window_size=10, detection_sigma=60)