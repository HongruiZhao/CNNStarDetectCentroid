import pandas as pd
import math
import numpy as np
from scipy.integrate import quad
from . import attitude
import cv2
import glob 

class star_img_sim:


    def __init__(self, camera_noise_flag, dark_frames_dir):
        """
            @param camera_noise_flag: if true, use camera noise. if false, use gaussian noise
            @param dark_frames_dir:
        """

        ### (1) Parameters for MT9V022 setup
        self.f = 16  # focal length, mm
        self.pixel_size = 6/1000.0 # size of a single pixel, mm
        self.height = int(480) # height of the image sensor, pixels
        self.width = int(640) # width of the image sensor, pixels

        self.diagnonal = math.sqrt( (0.5*self.pixel_size*self.height)**2 +  (0.5*self.pixel_size*self.width)**2 ) 
        self.fov =  2*math.atan2(self.diagnonal, self.f) # diagnonal FOV, rad 
        self.principal_point_U = 0 # principal point offset along image plane U axis, mm
        self.principal_point_V = 0 # principal point offset along image plane V axis, mm

        # coordinate of the intersection of the boresight and the focal plane under the focal plane frame, unit mm
        self.u0 = int(self.width / 2) * self.pixel_size + self.principal_point_U
        self.v0 = int(self.height / 2) * self.pixel_size + self.principal_point_V

        # for stellar aberration
        self.c = 299792458.0 # m/s. speed of light.

        # photon simulation and defocusing
        self.radius = 16.0 # radius of the aperture in mm (16mm for V-4416.0-1.2-HR )
        self.A = math.pi * (self.radius/1000)**2 # area of aperture, m^2, 
        self.epsilon = 0.8 # the average optical efficiency factor over V band
        self.E = 1 # an average correction factor for extinction due to the airmass over the V band
        self.delta_lambda = 0.089 # FWHM bandwidth of the V band. µm
        self.Q = 0.3 # average quantum efficiency of the image sensor over the V band
        self.T = 0.1 # exposure time, seconds
        self.sigma_psf = 0.85 # Gaussian radius of defcoused image, pixles
        self.half_width = 5 #int(3*self.sigma_psf + 0.5) # half width of the rectangular window used to create star images

        # for monte carlo sim 
        self.rng = np.random.default_rng()
        self.sigma_psf_min = 0.5
        self.sigma_psf_max = 2
        self.T_min = 0.1 # in seconds
        self.T_max = 1
        self.f_min = self.f - (self.f/100)
        self.f_max = self.f + (self.f/100)

        # Analog-to-digital
        # https://cs184.eecs.berkeley.edu/cs184_sp16_content/lectures/18_sensor/18_sensor_slides.pdf
        self.FWC = 8500 # electrons, full well capacity. 8500 for Aptina MT9P031 
        self.n_bits = 8 # bits of analog-to-digital converter

        # noises
        # table 5 from "An Accurate and Efficient Gaussian Fit Centroiding Algorithm for Star Trackers"
        self.dark_current = self.FWC/25 # electrons/pix/sec at current temperature. 25 is for Aptina MT9P031 at 55 degree celsius. 
        self.read_noise = self.FWC/30 #  electrons/pixel. 25 for MT9V022

        # read star catalog 
        name=['X','Y','Z','Magnitude','HIP ID']
        csv_file = pd.read_csv('./star_catalog/star_catalog_6.csv', header = 0, names = name)
        self.star_catalog = csv_file.values.tolist() 

        # star smears
        self.dt = 0.005 # exposure time for sub images. second. 

        # randomly pick a dark frame as the background noise
        self.dark_frames_dir = dark_frames_dir
        self.camera_noise_flag = camera_noise_flag
        self.num_of_frames = len(glob.glob(dark_frames_dir + "/*.npy")) # number of dark frames available
        print(f'num of dark frames currently in folder = {self.num_of_frames}')
        if self.camera_noise_flag:
            self.rg = np.random.default_rng()




    def set_camera_parameters(self, exposure, sigma_psf, f):
        """
            set camera parameters  
            @param exposure: exposure time, seconds  
            @param sigma_psf: Gaussian radius of defcoused image, pixles  
            @param f: focal length
        """
        # update exposure time 
        self.T = exposure
        # update sigma for Gaussian PSF
        self.sigma_psf = sigma_psf
        self.half_width = int(3*self.sigma_psf + 0.5) # half width of the rectangular window used to create star images
        # update focal length and fov
        self.f = f
        self.fov = 2*math.atan2(self.diagnonal, self.f)




    def get_real_centroid(self, A_BI, v_I):
        """
            calculate star real centroid coordinates (pixels) on the focal plane  
            @param A_BI: rotation matrix from  
            @param v_I: m/s. instantaneous velocity of the star tracker at the time the star is detected with respect to the ICRF represented under the ICRF  
            @return centroid_mm: star image real centroid coordinates (mm) under the focal plane frame, 2D coordinates + 1 mag + HIP ID   
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
            doot_product = np.dot(s_apparent_B, boresight)
            norm = np.linalg.norm(s_apparent_B) * np.linalg.norm(boresight)
            theta = math.acos( doot_product/ norm)

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
                    centroid_mm.append([u, v, self.star_catalog[i][3], self.star_catalog[i][4] ]) 
        

        return centroid_mm
    



    def defcousing(self, centroid_mm, exp):
        """
            defocusing the photons from starlight using PSF function  
            also calculate photons and convert them to electrons, and finally converted to bits   
            @param centroid_mm: star image real centroid coordinates (mm) under the focal plane frame, 2D coordinates + 1 mag + HIP ID  
            @param exp: exposure time, seconds
            @return star_image_raw: raw star image without sensor noises
        """

        def gaussian(uv_i, sigma_psf, mu):
            """
                A Gaussian PSF shapes in the u or v direction along the image plane 
            """
            g = 1/(math.sqrt(2*math.pi)*sigma_psf) * math.exp( -(uv_i - mu)**2 / (2*sigma_psf**2) )
            return g


        star_image_raw = np.zeros((self.height, self.width))
        for i in range( len(centroid_mm) ):
            
            u = centroid_mm[i][0]
            v = centroid_mm[i][1]
            
            # real star centroid in pixels
            mu_u = u / self.pixel_size
            mu_v = v / self.pixel_size

            # which pixel the real star centroid is located
            u_p = int(u // self.pixel_size)
            v_p = int(v // self.pixel_size)

            mag = centroid_mm[i][2]

            # the average rate of arrival of photons detected by the image sensor over the V band
            # Merline, W.K., Howell, S.B., "A  realistic model for point-sources imaged on array detectors: The model and initial results,” Experimental Astronomy, volume 6,163–210, 1995
            e_v = 1.085356 * 1e11 * 10**(-mag/2.5)
            n_avg = e_v * self.A * self.epsilon * self.E * self.delta_lambda

            # defocus star
            # Tjorven et al., "An Accurate and Efficient Gaussian Fit Centroiding Algorithm for Star Trackers," the Journal of the Astronautical Sciences, volume 6, 60-84, 2014.
            for near_u in range (-self.half_width, self.half_width+1, 1):
                for near_v in range(-self.half_width, self.half_width+1, 1):
                    
                    u_i = (u_p + near_u)
                    v_i = (v_p + near_v)

                    int_u = quad( gaussian, u_i-0.5, u_i+0.5, args=(self.sigma_psf, mu_u) ) # integration along u direction 
                    int_v = quad( gaussian, v_i-0.5, v_i+0.5, args=(self.sigma_psf, mu_v) ) # integration along v direction 
                    g_u_v = int_u[0] * int_v[0]
                    
                    if u_i >= 0 and u_i < self.width and v_i >= 0 and v_i < self.height:
                        # average rate of arrival of photons at a pixel
                        n_avg_pixel = n_avg * g_u_v 
                        # photon actually received by a pixel. L ́ena, P., “Observational Astrophysics 3rd Edition,” Springer, 2012
                        photons = self.rng.poisson(n_avg_pixel*exp)
                        # convert to electrons
                        star_image_raw[ v_i, u_i ] = photons * self.Q

        return star_image_raw




    def gaussian_noise(self, star_image_raw):
        """
            add dark current, and read noise as Gaussian noises
            @param star_image_raw: raw star image
            @return star_image_noise: star image with noises
        """
        star_image_noise = np.zeros((self.height, self.width))
        for col in range( self.width ):
            for row in range( self.height ):
                star_image_noise[row, col] = star_image_raw[row, col] + self.dark_current*self.T + self.read_noise * np.random.rand(1)
        return star_image_noise

    


    def adc(self, star_image):
        """
            do Analog-to-Digital converter    
            Ref: Tjorven et al., ”An Accurate and Efficient Gaussian Fit Centroiding Algorithm forStar Trackers,” the Journal of the Astronautical Sciences, volume 6, 60-84, 2014.5  
            @param star_image  
            @return star_image_dig: star image in digital count 0 ~ 2^bits - 1
        """
        star_image_dig = np.zeros((self.height, self.width))
        for col in range( self.width ):
            for row in range( self.height ):
                # get clean electron counts 
                pixel_value = star_image[row, col]
                # check FWC saturation, see note "simulate a star image" Eq6
                if pixel_value > self.FWC :
                    pixel_value = self.FWC
                if pixel_value < 0:
                    pixel_value = 0
                # quantization, results in digital counts, see note "simulate a star image" Eq6
                n_level = 2**self.n_bits - 1
                pixel_value = math.floor(pixel_value * n_level / self.FWC)

                star_image_dig[row, col] = pixel_value
        
        return star_image_dig




    def generate_star_image(self, q_input, v_I, T):
        """
            call functions to generate a star image  
            @param q_input : quaternion represent rotation from ICRF to body. q[3] is scalar  
            @param v_I: m/s. instantaneous velocity of the star tracker at the time the star is detected with respect to the ICRF represented under the ICRF  
            @param T: exposure time, seconds
            @return star_image_raw: simulated star image without noises
            @return centroid_mm: star image real centroid coordinates (mm) under the focal plane frame, 2D coordinates + 1 mag + HIP ID   
        """
        A_BI = attitude.q_to_a(q_input)
        centroid_mm = self.get_real_centroid(A_BI, v_I)
        star_image_raw = self.defcousing(centroid_mm, T)
        
        return star_image_raw, centroid_mm




    def generate_distance_map(self, centroid_mm):
        """
            generate a distance map. 
            @param centroid_mm: star image real centroid coordinates (mm) under the focal plane frame, [u, v, mag, HIP ID] .    
            @return dist_map: a map containing in each point its distance to the closet star centroid. in pixels  
        """

        dist_map = np.zeros((self.height, self.width))

        for current_row in range(self.height):
            for current_col in range(self.width):
                p_u = (current_col + 0.5) * self.pixel_size
                p_v = (current_row + 0.5) * self.pixel_size
                d_p_q = []

                for i in range(len(centroid_mm)):
                    q_u = centroid_mm[i][0]
                    q_v = centroid_mm[i][1]
                    dist =  math.sqrt((p_u-q_u)**2 + (p_v-q_v)**2) # Euclidean distance 
                    d_p_q.append(dist)
                d_p_q = np.asarray(d_p_q)
                
                # find shortest distance
                a = np.argmin(d_p_q)
                dist_map[current_row, current_col] = d_p_q[a] / self.pixel_size
        
        return dist_map





    def noise_from_camera(self):
        """
            randomly pick a dark frame from the folder
            @return real_noise: dark image from the camera
        """
        frame_id = self.rg.integers(0, self.num_of_frames)
        dark_frame = np.load(self.dark_frames_dir + f'/frame_{frame_id}.npy')

        # random 180 deg rotation
        rotate_flag = self.rg.integers(0, 2)
        if rotate_flag == 0:
            dark_frame = cv2.rotate(dark_frame, cv2.ROTATE_180)

        # random flip
        flip_flag = self.rg.integers(0, 4)
        if flip_flag == 0:
            dark_frame = cv2.flip(dark_frame, 0) # vertical flip
        elif flip_flag == 1:
            dark_frame = cv2.flip(dark_frame, 1) # horizontal flip
        elif flip_flag == 2:
            dark_frame = cv2.flip(dark_frame, -1) # flip both axes 

        return dark_frame




    # TODO
    def star_smear_segmentation(self, q, v_I, w, monte_carlo):
        """
            generate multiple subimages and combine them to simulate star _smear.
            generate binary segementation map of stars vs background for ML.    
            @param q : quaternion represent rotation from ICRF to body. q[3] is scalar  
            @param v_I: m/s. instantaneous velocity of the star tracker at the time the star is detected with respect to the ICRF represented under the ICRF  
            @param w: angular velocity. w_B^BI. rad/s.  
            @param monte_carlo: if true run monte carlo sim.  
            @return star_image_noise: simulated noisy star image
            @return gt: binary segmentation map of stars vs background
            @return T: exposure time in seconds
        """
        # set up monte carlo sim
        if monte_carlo:
            # fixed T at 100ms for now
            #self.T = self.rng.uniform(self.T_min, self.T_max) #
            self.sigma_psf = self.rng.uniform(self.sigma_psf_min, self.sigma_psf_max)
            self.half_width = int(3*self.sigma_psf + 0.5) # half width of the rectangular window used to create star images
            # update focal length and fov
            self.f = self.rng.uniform(self.f_min, self.f_max)
            self.fov = 2*math.atan2(self.diagnonal, self.f)


        number_of_img = int(self.T / self.dt)
        combined_image = np.zeros((self.height, self.width))
        for i in range(number_of_img):
            sub_image = self.generate_star_image(q, v_I, self.dt)
            combined_image = combined_image + sub_image
            q = attitude.rk4_kin(self.dt, w, q, 'quat')


        # analog to digital
        star_image_dig = self.adc(combined_image)

        # binary segmentation ground truth
        gt = (star_image_dig > 0)

        if self.camera_noise_flag:
            # get noise from real camera image
            real_noise = self.noise_from_camera()
            # final image
            star_image_noise = star_image_dig + real_noise
            star_image_noise =  np.clip(star_image_noise, 0, 255) # clip to 0~255

        else:
            # add gaussian noise
            star_image_noise = self.gaussian_noise(combined_image)
            # analog to digital
            star_image_noise = self.adc(star_image_noise)

        return star_image_noise, gt, self.T



    # TODO
    def star_smear_shortest_distance(self, q, v_I, w, monte_carlo):
        """
            generate multiple subimages and combine them to simulate star _smear.  
            generate shortest distance transformation map of centroids as groundtruth for ML training.     
            @param q : quaternion represent rotation from ICRF to body. q[3] is scalar  
            @param v_I: m/s. instantaneous velocity of the star tracker at the time the star is detected with respect to the ICRF represented under the ICRF  
            @param w: angular velocity. w_B^BI. rad/s.  
            @param monte_carlo: if true run monte carlo sim.  
            @return star_image_noise: simulated noisy star image
            @return gt: shortest distance transformation graph of centroids.  
        """
        # set up monte carlo sim
        if monte_carlo:
            # fixed T at 100ms for now
            #self.T = self.rng.uniform(self.T_min, self.T_max) #
            self.sigma_psf = self.rng.uniform(self.sigma_psf_min, self.sigma_psf_max)
            self.half_width = int(3*self.sigma_psf + 0.5) # half width of the rectangular window used to create star images
            # update focal length and fov
            self.f = self.rng.uniform(self.f_min, self.f_max)
            self.fov = 2*math.atan2(self.diagnonal, self.f)


        number_of_img = int(self.T / self.dt)
        combined_image = np.zeros((self.height, self.width))


        # first star image and its shortest distance transform 
        sub_image, gt = self.generate_star_image_sdt(q, v_I, self.dt)
        combined_image = combined_image + sub_image

        # generate the remaining star images 
        for i in range(number_of_img-1):
            q = attitude.rk4_kin(self.dt, w, q, 'quat')
            sub_image = self.generate_star_image(q, v_I, self.dt)
            combined_image = combined_image + sub_image


        if self.camera_noise_flag:
            # analog to digital
            star_image_dig = self.adc(combined_image)
            # get noise from real camera image
            real_noise = self.noise_from_camera()
            # final image
            star_image_noise = star_image_dig + real_noise
            star_image_noise =  np.clip(star_image_noise, 0, 255) # clip to 0~255
        else:
            # add gaussian noise
            star_image_noise = self.gaussian_noise(combined_image)
            # analog to digital
            star_image_noise = self.adc(star_image_noise)


        return star_image_noise, gt




    def no_smear(self, q, v_I, monte_carlo):
        """
            generate a single star image (no smear), the distance map, the segmentation map, and the centroids.  
            @param q : quaternion represent rotation from ICRF to body. q[3] is scalar  
            @param v_I: m/s. instantaneous velocity of the star tracker at the time the star is detected with respect to the ICRF represented under the ICRF  
            @param monte_carlo: if true run monte carlo sim.  
            @return star_image_noise: simulated noisy star image
            @return dist_map: distance map.
            @return seg_map: segmentation map. 
            @return centroid_mm: star image real centroid coordinates (mm) under the focal plane frame, 2D coordinates + 1 mag + HIP ID   
        """
        # set up monte carlo sim
        if monte_carlo:
            self.T = self.rng.uniform(self.T_min, self.T_max) 

            self.sigma_psf = self.rng.uniform(self.sigma_psf_min, self.sigma_psf_max)
            #self.half_width = 5 #int(3*self.sigma_psf + 0.5) # half width of the rectangular window used to create star images

            # update focal length and fov
            self.f = self.rng.uniform(self.f_min, self.f_max)
            self.fov = 2*math.atan2(self.diagnonal, self.f)
        
        star_img_raw, centroid_mm = self.generate_star_image(q, v_I, self.T)
        dist_map = self.generate_distance_map(centroid_mm)

        # binary segmentation ground truth
        seg_map = (star_img_raw > 0)    

        if self.camera_noise_flag:
            # analog to digital
            star_image_dig = self.adc(star_img_raw)
            # get noise from real camera image
            real_noise = self.noise_from_camera()
            # final image
            star_image_noise = star_image_dig.astype('float64') + real_noise.astype('float64')
            star_image_noise =  np.clip(star_image_noise, 0, 255) # clip to 0~255
        else:
            # add gaussian noise
            star_image_noise = self.gaussian_noise(star_img_raw)
            # analog to digital
            star_image_noise = self.adc(star_image_noise)


        return star_image_noise, dist_map, seg_map, centroid_mm










