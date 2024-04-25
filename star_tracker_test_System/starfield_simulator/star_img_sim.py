import pandas as pd
import math
import numpy as np
import attitude
import cv2


class star_img_sim:


    def __init__(self):

        ### optical simulator setup
        self.f = 350  # focal length, mm
        self.pixel_size = 0.05765109021605092 # size of a single pixel, mm
        self.height = int(1080) # height of the image sensor, pixels
        self.width = int(1920) # width of the image sensor, pixels
    
        self.diagnonal = math.sqrt( (0.5*self.pixel_size*self.height)**2 +  (0.5*self.pixel_size*self.width)**2 ) 
        self.fov =  2*math.atan2(self.diagnonal, self.f) # diagnonal FOV, rad 
        self.principal_point_U = 0 # principal point offset along image plane U axis, mm
        self.principal_point_V = 0 # principal point offset along image plane V axis, mm

        # coordinate of the intersection of the boresight and the focal plane under the focal plane frame, unit mm
        self.u0 = int(self.width / 2) * self.pixel_size + self.principal_point_U
        self.v0 = int(self.height / 2) * self.pixel_size + self.principal_point_V

        # read star catalog 
        name=['X','Y','Z','Magnitude','HIP ID']
        csv_file = pd.read_csv('./star_catalog/star_catalog_6.csv', header = 0, names = name)
        self.star_catalog = csv_file.values.tolist() 
        self.star_catalog_array = np.asarray(self.star_catalog)

        # for stellar aberration
        self.c = 299792458.0 # m/s. speed of light.




    def get_real_centroid(self, A_BI, v_I):
        """
            calculate star real centroid coordinates (pixels) on the focal plane  
            @param A_BI: rotation matrix from  
            @param v_I: m/s. instantaneous velocity of the star tracker at the time the star is detected with respect to the ICRF represented under the ICRF  
            @return centroid_mm: star image real centroid coordinates (mm) under the focal plane frame, 2D coordinates + 1 mag + HIP ID  + catalog id
        """

        # initialization
        centroid_mm = []

        boresight = np.array([0,0,1]) # star tracker boresight under its body frame

        for i in range( len(self.star_catalog) ):
            # true star vector under ICRF
            s_true_I = np.array([ self.star_catalog[i][0], self.star_catalog[i][1], self.star_catalog[i][2] ])

            # apparent star vector under ICRF after stellar aberration
            # Markley, L., and Crassidis, J., "Fundamentals of Spacecraft Attitude Determination and Control," Springer, 2014. (4.11)
            s_apparent_I = s_true_I + (v_I/self.c)
            s_apparent_I = s_apparent_I / np.linalg.norm(s_apparent_I)

            # apparent star vector under star tracker body frame
            s_apparent_B = A_BI @ s_apparent_I
            
            # check FOV 
            dot_product = np.dot(s_apparent_B, boresight)
            norm = np.linalg.norm(s_apparent_B) * np.linalg.norm(boresight)
            theta = math.acos( dot_product/ norm)

            if theta < self.fov: 
            # if theta < math.radians(5): 
            # if theta > math.radians(7.4) and theta < math.radians(7.5): # check error

                # get real centroid coordinates in mm under UV coordinate 
                # Zhao, H., "DEVELOPMENT OF A LOW-COST MULTI-CAMERA STAR TRACKER FOR SMALL SATELLITES," MS thesis, 2020. (Figure 3.2)(Eq 4.3) 
                u = self.u0 - s_apparent_B[0]*self.f/s_apparent_B[2]
                v = self.v0 - s_apparent_B[1]*self.f/s_apparent_B[2]
              
                # check if within image
                if u >= 0 and u < self.width*self.pixel_size and v >= 0 and v < self.height*self.pixel_size :

                    # Save centroids in mm + mag + HIP ID
                    centroid_mm.append([u, v, self.star_catalog[i][3], self.star_catalog[i][4], i ]) 
        

        return centroid_mm
    



    def get_real_centroid_vectorization(self, A_BI, v_I):
        """
            vectorization version of the "get_real_centroid" function to make it run faster 
            @param A_BI: rotation matrix from  
            @param v_I: m/s. instantaneous velocity of the star tracker at the time the star is detected with respect to the ICRF represented under the ICRF  
            @return centroid_mm: star image real centroid coordinates (mm) under the focal plane frame, 2D coordinates + 1 mag + HIP ID  + catalog id
        """
        boresight = np.array([0,0,1])# star tracker boresight under its body frame

        # shape = (5044, 3) for mag 6 catalog. each row is a star vector under ICRF
        s_true_I = self.star_catalog_array[:,0:3]

        # apparent star vector under ICRF after stellar aberration
        # Markley, L., and Crassidis, J., "Fundamentals of Spacecraft Attitude Determination and Control," Springer, 2014. (4.11)
        # v_I/self.c has shape (3,), match the number of columns of s_true_I, thus it is broadcastable
        aberration = (v_I/self.c).reshape(1,3)
        s_apparent_I = s_true_I + aberration

        row_sum = np.sum(s_apparent_I*s_apparent_I, axis=1, keepdims=True)
        row_sum = np.sqrt(row_sum)
        s_apparent_I = s_apparent_I / row_sum

        # each column is an apparent star vector under star tracker body frame 
        s_apparent_B = A_BI @ np.transpose(s_apparent_I)        

        # get real centroid coordinates in mm under UV coordinate 
        # Zhao, H., "DEVELOPMENT OF A LOW-COST MULTI-CAMERA STAR TRACKER FOR SMALL SATELLITES," MS thesis, 2020. (Figure 3.2)(Eq 4.3) 
        x_B = s_apparent_B[0,:]
        y_B = s_apparent_B[1,:]
        z_B = s_apparent_B[2,:]
        u = self.u0 - self.f*(x_B/z_B)
        v = self.v0 - self.f*(y_B/z_B)

        # check if within image
        within_image = (u >= 0) * (u < self.width*self.pixel_size) * (v >= 0) * (v < self.height*self.pixel_size) * (z_B > 0)

        # store centroid ( u(mm), v(mm), mag, HIPID)
        centroid_mm = []
        for index in np.nonzero(within_image)[0]:
           centroid_mm.append([u[index], v[index], self.star_catalog_array[index, 3], self.star_catalog_array[index, 4], index ])

        return centroid_mm




    def generate_star_img_op_sim(self, q_input, calib_flag, radius):
        """
            generate a star image for star tracker optical simulator. no noise, no defocusing, a star is represented by a single pixel.\\
            @param q_input : quaternion represent rotation from ICRF to body. q[3] is scalar.\\ 
            @param calib_flag: if true, display a small circle around the center of the screen. for calibration.\\
            @param radius: radius of the circle. pixels.\\
            @return star_img_op_sim.
            @return centroid_mm
        """

        star_img_op_sim = np.zeros((self.height, self.width))

        if calib_flag:
            u_p = int(self.u0 // self.pixel_size)
            v_p = int(self.v0 // self.pixel_size)
            # the center coordinates of circle.
            # (column, row)
            center = (u_p, v_p) 
            # for BGR, color as a tuple, eg: (255,0,0) for blue. For grayscale, just pass the scalar value.
            color_gray = 255
            thickness = -1 # Using thickness of -1 px to fill the circle
            # directly draw on input image 
            cv2.circle(star_img_op_sim, center, radius, color_gray, thickness)
            cv2.line(star_img_op_sim, (0,int(self.height/2)), (int(self.width),int(self.height/2)), (255,255,255), 1) # color in BGR 
            cv2.line(star_img_op_sim, (int(self.width/2),0), (int(self.width/2),int(self.height)), (255,255,255), 1) 
            
            centroid_mm = None

        else:
            A_BI = attitude.q_to_a(q_input)
            #centroid_mm = self.get_real_centroid(A_BI, np.array([0,0,0]))
            centroid_mm = self.get_real_centroid_vectorization(A_BI, np.array([0,0,0]))

            # # Only show selected stars 
            # centroid_all = self.get_real_centroid(A_BI, np.array([0,0,0]))
            # centroid_mm = []
            # centroid_mm.append(centroid_all[9])
            # centroid_mm.append(centroid_all[0])
            # centroid_mm.append(centroid_all[2])

            # import pandas as pd
            # name=['U (mm)','V (mm)','Mag', 'HID']
            # test=pd.DataFrame(columns=name, data=centroid_mm)
            # test.to_csv('./testcsv_4.csv',encoding='gbk')

            for i in range( len(centroid_mm) ):
                
                u = centroid_mm[i][0]
                v = centroid_mm[i][1]

                # which pixel the real star centroid is located
                u_p = int(u // self.pixel_size)
                v_p = int(v // self.pixel_size)

                star_img_op_sim[ v_p, u_p ] = 100

            
            star_img_op_sim = star_img_op_sim.astype('uint8')
        
        return star_img_op_sim, centroid_mm



    




