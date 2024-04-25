import math

ratio_w = 16 # width  
ratio_h = 9 # height  
diagonal = 5 # inches 
res_w = 1920 
res_h = 1080


diagonal = diagonal * 25.4 # mm

# (ratio_w**2 + ratio_h**2) * a**2  = d**2
a = math.sqrt(   diagonal**2 / (ratio_w**2 + ratio_h**2) )

width = a * ratio_w
height = a * ratio_h


pixel_size = width / res_w
print(f'pixel size = {pixel_size} mm') 
