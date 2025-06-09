import numpy as np
import math
import pandas as pd

class geometric_voting:
    """
        Michael et al., "Geometric Voting Algorithm for Star Trackers," in IEEE Transactions on
        Aerospace and Electronic Systems, vol. 44, no. 2, pp. 441-456, April 2008
    """
    # voting mehtod tolerance and threshold
    #tolerance = 0.035 # deg, 126 arcsec
    #tolerance = 0.05
    tolerance = 0.03
    max_vote = 2 # for verification


    #-------------------------------------------------------------------------------------------------------------
    def __init__( self ):

        # read star pair 
        name=['Angle','CatalogID1','CatalogID2'] # CatalogID = row index of a star at  the star catalog
        csv_file = pd.read_csv('./catalog_data/star_pair_6.1.csv', header = 0, names = name)
        self.star_pair = csv_file.values.tolist() 
        self.star_pair.sort( key = lambda col: col[0] ) # sorting, angle(first column) increasing order

        # read star catalog 
        name=['X','Y','Z','Magnitude','HIP ID']
        csv_file = pd.read_csv('./catalog_data/star_catalog_6.1.csv', header = 0, names = name)
        self.star_catalog = csv_file.values.tolist() 

        # For centroid
        self.centroid_result = [] # a list whose row element is [ centroid_x, centroid_y, sum_I ]

        # For star Identification 
        self.image_pair = [] # a list save all star pairs who are detected on the focal plane. row element [angle, imageID1, imageID2]
        self.star_vectors = [] # a list saving computed body frame star vectors from centroid points. row element [ s_x, s_y, s_z ]
        self.star_vectors_unver = [] # a list matched J2000 ECI frame catalog star vectors from centroid points. unverified. row element [ x, y, z, catalog_id ]
        self.star_vectors_ver = [] # a list matched J2000 ECI frame catalog star vectors from centroid points. verified. row element [ x, y, z, catalog_id, centroid_id, votes ]


    #-------------------------------------------------------------------------------------------------------------
    def find_image_pair(self, angular_distance_array, comb):
        for i in range(comb.shape[0]):
            id1 = comb[i,0]
            id2 = comb[i,1]
            angular_distance = angular_distance_array[id1, id2]
            self.image_pair.append( (angular_distance, id1, id2) )


    #-------------------------------------------------------------------------------------------------------------
    def binary_search( self, arr, start, rear, angle, voteResults, imageID1, imageID2, tolerance ): 
        ### while start ###
        end = rear # last element
        while start <= rear : 
            mid = start + (rear - start)/2
            mid = int(mid) # get rid of decimal directly

            # Check if x is present at mid 
            if abs(arr[mid][0] - angle) < tolerance: 
                # get votes
                arrID1 = int(arr[mid][1])
                arrID2 = int(arr[mid][2])
                voteResults[imageID1, arrID1] = 1
                voteResults[imageID1, arrID2] = 1
                voteResults[imageID2, arrID1] = 1
                voteResults[imageID2, arrID2] = 1

                ## find more match angle ##
                # check left first 
                flag = 1 
                while ( (mid - flag) >= start ) and abs(arr[mid - flag][0] - angle) < tolerance :
                    arrID1 = int(arr[mid - flag][1])
                    arrID2 = int(arr[mid - flag][2])
                    voteResults[imageID1, arrID1] = 1
                    voteResults[imageID1, arrID2] = 1
                    voteResults[imageID2, arrID1] = 1
                    voteResults[imageID2, arrID2] = 1
                    flag += 1
                # check right then 
                flag = 1 
                while ( (mid + flag) <=  end ) and abs(arr[mid + flag][0] - angle) < tolerance :
                    # get votes
                    arrID1 = int(arr[mid + flag][1])
                    arrID2 = int(arr[mid + flag][2])
                    voteResults[imageID1, arrID1] = 1
                    voteResults[imageID1, arrID2] = 1
                    voteResults[imageID2, arrID1] = 1
                    voteResults[imageID2, arrID2] = 1
                    flag += 1
                # break    
                break
        
            # If x is greater, ignore left half 
            elif angle - arr[mid][0] > tolerance: 
                start = mid + 1
    

            # If x is smaller, ignore right half 
            else: 
                rear = mid - 1

        ### while end ###

        return voteResults


    #-------------------------------------------------------------------------------------------------------------
    def voting_method(self):

        # get constants
        star_number = len(self.star_catalog)
        centroid_number = len(self.centroid_result)
        star_pair_number = len(self.star_pair) 
        image_pair_number = len(self.image_pair)

        # detected centroid is linked to one of the star from catalog.
        # votes under each catalog star represent the probability that the detected centroid IS this certain catalog star 
        shape = (centroid_number, star_number)
        vote_results = np.zeros(shape)

        # save assign catalog star ID for detected centroids
        id_assign = np.zeros(centroid_number)

        ### start voting process ###
        for image_pair_id in range(image_pair_number):
            tmp_vote_results = np.zeros(shape)
            image_angle = self.image_pair[image_pair_id][0]
            image_id1 = self.image_pair[image_pair_id][1] # image id = row index of a centroid at centroid_result
            image_id2 = self.image_pair[image_pair_id][2]
            tmp_vote_results = self.binary_search(self.star_pair, 0, star_pair_number-1, image_angle, tmp_vote_results, int(image_id1), int(image_id2), self.tolerance )
            vote_results += tmp_vote_results
        ### voting process end ### 

        # DEBUG 
        # vote_csv = pd.DataFrame(data = vote_results)
        # vote_csv.to_csv('./vote_result.csv')

        # get highest vote catalog star
        for row in range(centroid_number):
            max_id = 0 # catalog star id that gets most vote
            for col in range(star_number):
                if vote_results[row, col] > vote_results[row, max_id]:
                    max_id = col
            id_assign[row] = max_id

        # get a non-verified inertial star vectors list
        for row in range(centroid_number):
            max_id = int(id_assign[row])
            self.star_vectors_unver.append([ self.star_catalog[max_id][0], self.star_catalog[max_id][1], self.star_catalog[max_id][2], max_id ])


    #-------------------------------------------------------------------------------------------------------------
    def verification_voting(self):
        # get constants
        vectors_number = len(self.centroid_result)
        image_pair_number = len(self.image_pair)

        # save vote results from each unverified star vector
        vote_results = np.zeros(vectors_number)

        ### start voting verification process ###
        for row in range(image_pair_number):
            # angle from image pairs
            angle_image = self.image_pair[row][0]
            id_1 = self.image_pair[row][1]
            id_2 = self.image_pair[row][2]

            # angle from unverified star vectors
            x_id_1 = self.star_vectors_unver[id_1][0]
            y_id_1 = self.star_vectors_unver[id_1][1]
            z_id_1 = self.star_vectors_unver[id_1][2]
            vector_id_1 = np.array([ x_id_1, y_id_1, z_id_1 ])

            x_id_2 = self.star_vectors_unver[id_2][0]
            y_id_2 = self.star_vectors_unver[id_2][1]
            z_id_2 = self.star_vectors_unver[id_2][2]
            vector_id_2 = np.array([ x_id_2, y_id_2, z_id_2 ])

            dot_product = np.dot(vector_id_1, vector_id_2)
            mag_1_2 = np.linalg.norm(vector_id_1) * np.linalg.norm(vector_id_2)
            dot_norm = dot_product / mag_1_2
            if dot_norm > 1:
                dot_norm = 1
            angle_unver = math.degrees( math.acos(dot_norm)  )

            # voting
            if abs(angle_image - angle_unver) < self.tolerance: 
                vote_results[id_1] += 1
                vote_results[id_2] += 1
        ### voting process end ###

        # get final star vectors 
        #DEBUG
        # print(f'verification voting result is {vote_results}')
        for row in range(vectors_number):
            if vote_results[row] >= self.max_vote:
                x = self.star_vectors_unver[row][0]
                y = self.star_vectors_unver[row][1]
                z = self.star_vectors_unver[row][2]
                catalog_id = self.star_vectors_unver[row][3]
                centroid_id = row
                hip_id = self.star_catalog[catalog_id][4]
                self.star_vectors_ver.append([x, y, z, catalog_id, centroid_id, vote_results[row], hip_id])


    #-------------------------------------------------------------------------------------------------------------
    def star_identification(self, angular_distance_array, comb):

        # clean previous data
        self.image_pair = [] # a list save all star pairs who are detected on the focal plane. row element [angle, imageID1, imageID2]
        self.star_vectors = [] # a list saving computed body frame star vectors from centroid points. row element [ s_x, s_y, s_z ]
        self.star_vectors_unver = [] # a list matched J2000 ECI frame catalog star vectors from centroid points. unverified. row element [ x, y, z, catalog_id ]
        self.star_vectors_ver = [] # a list matched J2000 ECI frame catalog star vectors from centroid points. verified. row element [ x, y, z, catalog_id, centroid_id, votes ]


        # star identification and attitude determination 
        if len(self.centroid_result) > 0:
            
            # star identification
            self.find_image_pair(angular_distance_array, comb)
            self.voting_method()
            self.verification_voting()

        else:
            print("No Star Detected")
            
    #------------------------------------------------------------------------------------------
    def catalog_pair(self):
        """
            generate all star pairs inside the star catalog.
        """
        diagnonal = math.sqrt( (0.5*self.pixel_size*self.image_height)**2 +  (0.5*self.pixel_size*self.image_width)**2 )
        circular_fov =  math.degrees( 2*math.atan2(diagnonal, self.f) )
        print(f'full circular fov is {circular_fov}')

        catalogTable = []
        catalogSize = len(self.star_catalog) # number of stars

        ### while start ###
        for catalogID1 in range(catalogSize-1): # star ID of the 2nd column
            Vector1 = np.array( [ self.star_catalog[catalogID1][0], self.star_catalog[catalogID1][1], self.star_catalog[catalogID1][2] ] )

            for catalogID2 in range(catalogID1+1, catalogSize, 1):
                Vector2 = np.array( [ self.star_catalog[catalogID2][0], self.star_catalog[catalogID2][1], self.star_catalog[catalogID2][2] ] )
                # get angle 
                dot_product = np.dot(Vector1, Vector2)
                mag_1_2 = np.linalg.norm(Vector1) * np.linalg.norm(Vector2)
                angle = math.degrees( math.acos(dot_product / mag_1_2) )
                if angle < circular_fov :
                    catalogTable.append( (angle, catalogID1, catalogID2) )

            # progress bar
            progress = "Catalog Progress = " + str(math.floor((catalogID1/catalogSize)*100)) + " %"
            print(progress, end="\r")

        ### while end ###


        return catalogTable


    #------------------------------------------------------------------------------------------
    def csv_pair(self): 
        """
            Write all star pairs into a csv file 
        """
        name=['Angle','CatalogID1','CatalogID2']
        catalogTable = self.catalog_pair()
        catalogTable.sort( key = lambda col: col[0] ) # sorting, angle(first column) increasing order
        CSVfile = pd.DataFrame( columns = name, data = catalogTable )
        CSVfile.to_csv('./star_pair_5.csv',encoding='gbk')

        return 0
    

 



