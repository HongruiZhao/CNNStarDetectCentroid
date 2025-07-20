import sys

import numpy
import catalog_functions
import math

"""
    generate a star catalog including every possible pair of stars within a certain magnitude range from hippacros
"""
Magnitude = 6.1 # highest magnitude of the star catalog 

# get all star vectos
stars = catalog_functions.catalog(Magnitude)
catalog_functions.csv_catalog(stars, Magnitude)
