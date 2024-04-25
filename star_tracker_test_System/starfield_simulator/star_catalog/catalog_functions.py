import numpy as np
import math
import pandas as pd

# skyfield
from skyfield.api import Star, load
from skyfield.data import hipparcos


def catalog(Magnitude):
    """
        find stars in hippacros within Magnitude. \\
        @param Magnitude: highest magnitude  \\
        @return : an array including all star vectors within Magnitude and HIP ID [ index, x, y, z, mag, HIP ID ]
    """
    # load hippacros star catalog
    # https://rhodesmill.org/skyfield/stars.html#the-hipparcos-catalog
    with load.open(hipparcos.URL) as f:
        df = hipparcos.load_dataframe(f) 

    # only keep stars that are brighter than `Magnitude`
    # https://rhodesmill.org/skyfield/stars.html#filtering-the-star-catalog
    df = df[df['magnitude'] <= Magnitude ]

    # convert to star objects
    bright_stars = Star.from_dataframe(df)

    # load earth
    planets = load('de421.bsp')
    earth = planets['earth']

    # get current epoch
    ts = load.timescale()
    now = ts.now()

    # the right ascension and the declination for current epoch, with respect the reference frame ICRF
    # this will reflect proper motion and parallax: https://rhodesmill.org/skyfield/stars.html#the-hipparcos-catalog 
    # https://rhodesmill.org/skyfield/stars.html#proper-motion-and-parallax
    # radec() return rad & dec under ICRS: https://rhodesmill.org/skyfield/api-position.html#skyfield.positionlib.ICRF.radec
    astrometric   = earth.at(now).observe(bright_stars)
    ra, dec, distance = astrometric.radec()

    ### get unit vector for each star ###
    # how many stars we have 
    n = len(df)

    # get unit vector for each stars
    s = (n,5) # n rows, 5 columns
    uv = np.zeros(s) # to save unit vectors and magnitude

    # for loop to get unit vectors
    for i in range(n):
        # Get RA Dec
        RA = ra.hours[i]*15 # deg
        RA = math.radians(RA) # rad
        Dec = dec.degrees[i] #deg
        Dec = math.radians(Dec) # rad

        # get unit vector
        # the equatorial frame - 4.1.2 - Léna “Observational Astrophysics”
        z = math.sin(Dec)
        x = math.cos(Dec) * math.cos(RA)
        y = math.cos(Dec) * math.sin(RA)

        # get magnitude
        Mag =  df.loc[:,['magnitude']] #use loc(pandas function) to extract values from 'magitude' column label
        Mag = Mag.iloc[i,[0]] #use iloc to extract values from row i column 0(only 0)

        # get hip id 
        hip = df.index[i]

        # save
        uv[i,0] = x
        uv[i,1] = y
        uv[i,2] = z
        uv[i,3] = Mag
        uv[i,4] = hip

    ######

    return uv


#------------------------------------------------------------------------------------------
def csv_catalog( stars, Magnitude ): 
    """
        Write all stars within magnitude range into a csv file  
    """
    name=['X','Y','Z', 'Magnitude', 'HIP ID']
    CSVfile = pd.DataFrame( columns = name, data = stars )
    CSVfile.to_csv('./star_catalog_{}.csv'.format(Magnitude),encoding='gbk')

    return 0


