import numpy as np
import math


def angular_distacne(vector_1, vector_2):

    norm = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    dot_product = np.dot(vector_1, vector_2)
    theta = math.degrees( math.acos(dot_product / norm) )
    return theta



def angular_distacne_from_centroid(centroid_1, centroid_2):

    focal_length = 15.98 # mm
    pixel_size = 0.0022*2 # mm

    vector_1 = np.array([ -centroid_1[0]*pixel_size, -centroid_1[1]*pixel_size, focal_length ])
    print(centroid_1)
    print(vector_1 / np.linalg.norm(vector_1))
    vector_2 = np.array([ -centroid_2[0]*pixel_size, -centroid_2[1]*pixel_size, focal_length ])

    norm = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    dot_product = np.dot(vector_1, vector_2)
    theta = math.degrees( math.acos(dot_product / norm) )
    return theta


def main():
    # star vector from catalog
    star_0 = [-0.11988, 0.049006, 0.991578]
    star_1 = [-0.09596, -0.08821, 0.991469]
    star_2 = [0.084674, -0.09862, 0.991516]

    theta = angular_distacne(star_0, star_1)
    print(f'the angular distance between star 0 and star 1 is {theta} deg')

    
    theta = angular_distacne(star_0, star_2)
    print(f'the angular distance between star 0 and star 2 is {theta} deg')

    
    theta = angular_distacne(star_1, star_2)
    print(f'the angular distance between star 1 and star 2 is {theta} deg')

    
    centroid_0 = [303.9, 366.3]
    centroid_1 = [-441.2, -186.5]
    centroid_2 = [-358.3, 320.8]

    theta = angular_distacne_from_centroid(centroid_0, centroid_1)
    print(f'the angular distance between centroid 0 and centroid 1 is {theta} deg')

    theta = angular_distacne_from_centroid(centroid_1, centroid_2)
    print(f'the angular distance between centroid 1 and centroid 2 is {theta} deg')

    theta = angular_distacne_from_centroid(centroid_2, centroid_0)
    print(f'the angular distance between centroid 2 and centroid 0 is {theta} deg')




    # star vector from aspire studio 
    star_0 = [-0.0830, 0.1, 0.9915]
    star_1 = [0.1203,  -0.0509, 0.9914]
    star_2 = [0.0978, 0.0876, 0.9913]
    
    theta = angular_distacne(star_0, star_1)
    print(f'the angular distance between star 0 and star 1 is {theta} deg')

    
    theta = angular_distacne(star_0, star_2)
    print(f'the angular distance between star 0 and star 2 is {theta} deg')

    
    theta = angular_distacne(star_1, star_2)
    print(f'the angular distance between star 1 and star 2 is {theta} deg')



    
if __name__ == "__main__":
    main()
    