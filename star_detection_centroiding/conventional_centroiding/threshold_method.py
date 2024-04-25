import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2



def detection_globalThreshold(image_input, factor=5, pixel_area=6):
    """
        flood fill + Liebe adaptive threshold  
        @param image_input: star image as a np array  
        @param factor: 
        @param pixel_area: minimum number of contiguous lit pixels for a valid star detection. 6 in ST16
        @return binary_mask: 1 = star centroid at this pixel, 0 = no star centroid
    """
   
    neighbours = [(-1,0), (0,1), (1,0), (0,-1)] # 4-connectivity
    image_height = image_input.shape[0]
    image_width = image_input.shape[1] 
    
    detection_limit = np.mean(image_input) + factor * np.std(image_input)  # Liebe adaptive threshold
    pixel_sum = detection_limit * ( pixel_area + 1 )
    
    binary_mask = np.zeros(image_input.shape)

    # find local average for each pixel
    image_mean = ndimage.uniform_filter(input=image_input, size=[1,129])

    for current_row in range(image_height):
        for current_col in range(image_width):
            if image_input[current_row, current_col] > detection_limit:
                flood_fill = [] 
                flood_fill.append([current_row, current_col]) 

                # save pixels of this region
                region = []

                ### flood fill begin ###
                # save pixel into region_centroid and delete it from flood_fill
                # save new neighbour into flood_fill
                # when there is no new neighbour and all pixels are deleted, quit
                while flood_fill:
                    row, col = flood_fill.pop()
                    Mag = image_input[row, col]
                    region.append([col, row, Mag]) # u-v coordinate( u = col index, v = row index) and magnitude of this pixel
                 
                    image_input[row,col] = 0 # avoid re-detection

                    for drow, dcol in neighbours:
                        nrow, ncol = row + drow, col + dcol
                        try:
                            # if current pixel is inside the image plane and above the threshold
                            if image_input[nrow, ncol] > detection_limit:
                                
                                    # to see if (nx,ny) already in the stack. avoid recomputing
                                    if [nrow, ncol] not in flood_fill:
                                        flood_fill.append([nrow, ncol])
                        except IndexError:
                            pass
                    
                    if len(region) > 40:
                        break
                ### flood fill end ###
                
                # check for blob size and total pixel value count 
                region = np.asarray( region )
                region_size = int(region.size/3)
                if region_size >= pixel_area: 
                    sum_I = 0
                    for pixel_count in range(region_size):
                        sum_I = sum_I + region[pixel_count, 2]
                    if sum_I > pixel_sum: # Integrated Intensity
                         
                        # find the brightest pixel 
                        col_bri = int(region[np.argmax(region[:, 2] ), 0])
                        row_bri = int(region[np.argmax(region[:, 2] ), 1])

                        binary_mask[int(row_bri), int(col_bri)] = 1

    return binary_mask





def detection_WITM(image_input, delta=-0.5, DELTA=0.2, pixel_area=6):
    """
        weighted iterative threshold method
        https://www.sciencedirect.com/science/article/pii/S0030402613002490  
        @param image_input: star image as a np array  
        @param delta: weight coefficient 
        @param DELTA: threshold for Ti+1 - Ti
        @param pixel_area: minimum number of contiguous lit pixels for a valid star detection. 6 in ST16
        @return binary_mask: 1 = star centroid at this pixel, 0 = no star centroid
    """
   
    neighbours = [(-1,0), (0,1), (1,0), (0,-1)] # 4-connectivity
    image_height = image_input.shape[0]
    image_width = image_input.shape[1] 
    
    T_old = (np.max(image_input) + np.min(image_input)) / 2
    
    # weighted iterative threshold method
    while True:
        R1 = []
        R2 = []
        for current_row in range(image_height):
            for current_col in range(image_width):
                if image_input[current_row, current_col] > T_old:
                    R1.append(image_input[current_row, current_col])
                else:
                    R2.append(image_input[current_row, current_col])
        R1 = np.asarray(R1)
        R2 = np.asarray(R2)
        mu_1 = np.mean(R1)
        mu_2 = np.mean(R2)
        T_new = ((1+delta)*mu_1 + (1-delta)*mu_2) / 2
        if abs(T_new - T_old) < DELTA:
            break
        else:
            T_old = T_new

    detection_limit = T_new
    pixel_sum = detection_limit * ( pixel_area + 1 )
   
    binary_mask = np.zeros(image_input.shape)

    for current_row in range(image_height):
        for current_col in range(image_width):
            if image_input[current_row, current_col] > detection_limit:
                flood_fill = [] 
                flood_fill.append([current_row, current_col]) 

                # save pixels of this region
                region = []

                ### flood fill begin ###
                # save pixel into region_centroid and delete it from flood_fill
                # save new neighbour into flood_fill
                # when there is no new neighbour and all pixels are deleted, quit
                while flood_fill:
                    row, col = flood_fill.pop()
                    Mag = image_input[row, col]
                    region.append([col, row, Mag]) # u-v coordinate( u = col index, v = row index) and magnitude of this pixel
                 
                    image_input[row,col] = 0 # avoid re-detection

                    for drow, dcol in neighbours:
                        nrow, ncol = row + drow, col + dcol
                        try:
                            # if current pixel is inside the image plane and above the threshold
                            if image_input[nrow, ncol] > detection_limit:
                                
                                    # to see if (nx,ny) already in the stack. avoid recomputing
                                    if [nrow, ncol] not in flood_fill:
                                        flood_fill.append([nrow, ncol])
                        except IndexError:
                            pass
                    
                    if len(region) > 40:
                        break
                ### flood fill end ###
                
                # check for blob size and total pixel value count 
                region = np.asarray( region )
                region_size = int(region.size/3)
                if region_size >= pixel_area: 
                    sum_I = 0
                    for pixel_count in range(region_size):
                        sum_I = sum_I + region[pixel_count, 2]
                    if sum_I > pixel_sum: # Integrated Intensity
                         
                        # find the brightest pixel 
                        col_bri = int(region[np.argmax(region[:, 2] ), 0])
                        row_bri = int(region[np.argmax(region[:, 2] ), 1])

                        binary_mask[int(row_bri), int(col_bri)] = 1

    return binary_mask




def detection_ST16(image_input, threshold=4.3, pixel_area=6, pixel_sum=50):
    """
        ST16 detection routine
        @param image_input: star image as a np array  
        @param threshold: > local average + threshold to be detected. 1.7% of total counts in ST16 
        @param pixel_area: minimum number of contiguous lit pixels for a valid star detection. 6 in ST16
        @param pixel_sum: minimum summed intensity of all contiguous lit pixels a candidate star should have. 20% in ST16
        @return binary_mask: 1 = star centroid at this pixel, 0 = no star centroid
    """

    neighbours = [(-1,0), (0,1), (1,0), (0,-1)] # 4-connectivity
    image_height = image_input.shape[0]
    image_width = image_input.shape[1] 
    
    binary_mask = np.zeros(image_input.shape)

    # find local average for each pixel
    image_mean = ndimage.uniform_filter(input=image_input, size=[1,129])

    for current_row in range(image_height):
        for current_col in range(image_width):
            if image_input[current_row, current_col] > (image_mean[current_row, current_col] + threshold):
                flood_fill = [] 
                flood_fill.append([current_row, current_col]) 

                # save pixels of this region
                region = []

                ### flood fill begin ###
                # save pixel into region_centroid and delete it from flood_fill
                # save new neighbour into flood_fill
                # when there is no new neighbour and all pixels are deleted, quit
                while flood_fill:
                    row, col = flood_fill.pop()
                    Mag = image_input[row, col]
                    region.append([col, row, Mag]) # u-v coordinate( u = col index, v = row index) and magnitude of this pixel
                 
                    image_input[row,col] = 0 # avoid re-detection

                    for drow, dcol in neighbours:
                        nrow, ncol = row + drow, col + dcol
                        try:
                            # if current pixel is inside the image plane and above the threshold
                            if image_input[nrow, ncol] > (image_mean[current_row, current_col] + threshold) :
                                
                                    # to see if (nx,ny) already in the stack. avoid recomputing
                                    if [nrow, ncol] not in flood_fill:
                                        flood_fill.append([nrow, ncol])
                        except IndexError:
                            pass
                    
                    if len(region) > 40:
                        break
                ### flood fill end ###
                
                # check for blob size and total pixel value count 
                region = np.asarray( region )
                region_size = int(region.size/3)
                if region_size >= pixel_area: 
                    sum_I = 0
                    for pixel_count in range(region_size):
                        sum_I = sum_I + region[pixel_count, 2]
                    if sum_I > pixel_sum: # Integrated Intensity
                         
                        # find the brightest pixel 
                        col_bri = int(region[np.argmax(region[:, 2] ), 0])
                        row_bri = int(region[np.argmax(region[:, 2] ), 1])

                        binary_mask[int(row_bri), int(col_bri)] = 1

    return binary_mask




def detection_erosion_dilation(image_input, gaussian_sigma, average_window_size, detection_sigma, pixel_area=6):
    """
        Ref: Motion-blurred star acquisition method of the star tracker under high dynamic conditions
        @param image_input: star image as a np array  
        @param gaussian_sigma: sigma for Gaussian filter. should be consistent with star point spread energy distribution
        @param average_window_size: the size of average filter window 
        @param detection_sigma: threshold = B + sigma. sigma = 10~20 according to the paper.
        @param pixel_area: minimum number of contiguous lit pixels for a valid star detection. 6 in ST16
        @return binary_mask: 1 = star centroid at this pixel, 0 = no star centroid
    """

    neighbours = [(-1,0), (0,1), (1,0), (0,-1)] # 4-connectivity
    image_height = image_input.shape[0]
    image_width = image_input.shape[1] 
    
    binary_mask = np.zeros(image_input.shape)

    # gaussian filter 
    # the size of kernel along each axis will be 2*radius + 1
    # radius is a new argument for gaussian_filter after scipy 1.10
    c = ndimage.gaussian_filter(image_input, sigma=gaussian_sigma, radius=1)

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
    T = B + detection_sigma

    for current_row in range(image_height):
        for current_col in range(image_width):
            if image_input[current_row, current_col] > T[current_row, current_col]:
                flood_fill = [] 
                flood_fill.append([current_row, current_col]) 

                # save pixels of this region
                region = []

                ### flood fill begin ###
                # save pixel into region_centroid and delete it from flood_fill
                # save new neighbour into flood_fill
                # when there is no new neighbour and all pixels are deleted, quit
                while flood_fill:
                    row, col = flood_fill.pop()
                    Mag = image_input[row, col]
                    region.append([col, row, Mag]) # u-v coordinate( u = col index, v = row index) and magnitude of this pixel
                 
                    image_input[row,col] = 0 # avoid re-detection

                    for drow, dcol in neighbours:
                        nrow, ncol = row + drow, col + dcol
                        try:
                            # if current pixel is inside the image plane and above the threshold
                            if image_input[nrow, ncol] > T[nrow, ncol]:
                            
                                    # to see if (nx,ny) already in the stack. avoid recomputing
                                    if [nrow, ncol] not in flood_fill:
                                        flood_fill.append([nrow, ncol])
                        except IndexError:
                            pass
                    
                    if len(region) > 40:
                        break
                ### flood fill end ###
                
                # check for blob size 
                region = np.asarray( region )
                region_size = int(region.size/3)
                if region_size >= pixel_area:  
                    # find the brightest pixel 
                    col_bri = int(region[np.argmax(region[:, 2] ), 0])
                    row_bri = int(region[np.argmax(region[:, 2] ), 1])

                    binary_mask[int(row_bri), int(col_bri)] = 1

    return binary_mask




def centroiding_CenterOfMass(image_input, mask_input, window_size):
    """
        center of mass centroiding 
        @param image_input: star image as a np array          
        @param mask_input: = 1 if star centroid is here, = 0 otherwise.
        @param window_size: (window_size*2+1) x (window_size*2+1) window around the brightest pixel to compute centroid 
        @return centroid_result: an array whose row element is [ centroid_u (mm), centroid_v (mm), sum_I (counts) ].
    """

    image_height = mask_input.shape[0]
    image_width = mask_input.shape[1] 
    pixel_size = 6/1000.0 # size of a single pixel, mm
    centroid_result = [] # a list whose row element is [ centroid_u, centroid_v, sum_I ]

    for current_row in range(image_height):
        for current_col in range(image_width):

             if mask_input[current_row, current_col] == 1:
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
        @return centroid_result: an array whose row element is [ centroid_u (mm), centroid_v (mm), sum_I (counts) ]
    """

    image_height = mask_input.shape[0]
    image_width = mask_input.shape[1] 
    pixel_size = 6/1000.0 # size of a single pixel, mm
    centroid_result = [] # a list whose row element is [ centroid_u, centroid_v, sum_I ]
    background_value = 10 # mean value of a dark frame to fill pixel value out of bound
    
    for current_row in range(image_height):
        for current_col in range(image_width):

             if mask_input[current_row, current_col] == 1:
                
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





def centroid_com(image_input, factor):
    """
        flood fill + Liebe adaptive threshold + center of mass method to find the centroid.  
        @param image_input: star image as a np array          
        @param factor: 
        @return centroid_result: an array whose row element is [ centroid_u, centroid_v, sum_I ]
    """

    neighbours = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    image_height = image_input.shape[0]
    image_width = image_input.shape[1] 
    pixel_size = 6/1000.0 # size of a single pixel, mm
    pixel_area = 4 # pixels, an region has to have larger area that this to be put into centroid calculation
    detection_limit = np.mean(image_input) + factor * np.std(image_input)  # Liebe adaptive threshold
    pixel_sum = detection_limit * ( pixel_area + 1 )

    centroid_result = [] # a list whose row element is [ centroid_u, centroid_v, sum_I ]


    for current_row in range(image_height):
        for current_col in range(image_width):

            if image_input[current_row, current_col] > detection_limit:
                flood_fill = [] 
                flood_fill.append([current_row, current_col]) 

                # save pixels of this region for centroid 
                region_centroid = []

                ### flood fill begin ###
                # save pixel into region_centroid and delete it from flood_fill
                # save new neighbour into flood_fill
                # when there is no new neighbour and all pixels are deleted, quit
                while flood_fill:
                    row, col = flood_fill.pop()
                    Mag = image_input[row, col]
                    pixel = [col, row, Mag] # u-v coordinate( u = col index, v = row index) and magnitude of this pixel
                    region_centroid.append(pixel) 

                    image_input[row,col] = 0 # avoid re-detection

                    for drow, dcol in neighbours:
                        nrow, ncol = row + drow, col + dcol
                        saveFlag = 1 # indicate whether to save this coordinate or not

                        # if current pixel is inside the image plane and above the threshold
                        if ( 0 <= nrow < image_height and 0 <= ncol < image_width and image_input[nrow, ncol] > detection_limit ):
                            # to see if (nx,ny) already in the stack. avoid recomputing
                            for test_row, test_col in flood_fill:
                                if nrow == test_row and ncol == test_col:
                                    saveFlag = 0
                                    break
                            if saveFlag == 1:
                                flood_fill.append([nrow, ncol])
            
                ### flood fill end ###
                
                region_centroid = np.asarray( region_centroid )
                region_size = int(region_centroid.size/3)

                if region_size > pixel_area: # blob size: at least 2x2 pixel area 
                    ### centroid computation ###
                    # Ref: A software package for evaluating the performance of a star sensor operation
                    u_sum_xI = 0
                    v_sum_yI = 0
                    sum_I = 0

                    for pixel_count in range( region_size ):
                        u_sum_xI = u_sum_xI + pixel_size * (region_centroid[pixel_count, 0] + 0.5) * region_centroid[pixel_count, 2] # pixel coordinate starts from 0, pick the center of a pixel 
                        v_sum_yI = v_sum_yI + pixel_size * (region_centroid[pixel_count, 1] + 0.5) * region_centroid[pixel_count, 2] 
                        sum_I = sum_I + region_centroid[pixel_count, 2]

                    if sum_I > pixel_sum: # Integrated Intensity
                        centroid_u = u_sum_xI / sum_I
                        centroid_v = v_sum_yI / sum_I
                        centroid_result.append([ centroid_u, centroid_v, sum_I ]) # save centroid coordinate and magnitude


    return centroid_result