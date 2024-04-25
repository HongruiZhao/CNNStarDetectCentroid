# Arducam sdk
support for python 3.8 are recently added. https://www.arducam.com/forums/topic/easy-python-programming-sample-for-usb2-camera-shield/

Arducam repository: https://github.com/ArduCAM/ArduCAM_USB_Camera_Shield

go to arducam_usb_camera_shield/Windows/Python/Streaming_demo/ to find all the .pyd files

for python 3.8, `import ArducamSDK` needs:
-`ArducamSDK.pyd`
-`ArducamSDK.cp38-win_amd64.pyd`


# op_sim_anime.py
Display an animated star field for star tracker optical test system.

# data_shortest_distance.py
generate star images and their corresponding shortest distance transform maps for ML training.


# star_img_sim.py
generate star image 

### (1) Parameters for MT9V022 setup
self.f = 16  # focal length, mm
self.pixel_size = 6/1000.0 # size of a single pixel, mm
self.height = int(480) # height of the image sensor, pixels
self.width = int(640) # width of the image sensor, pixels

### (2) optical simulator setup
self.f = 250  # focal length, mm
self.pixel_size = 0.05765109021605092 # size of a single pixel, mm
self.height = int(1080) # height of the image sensor, pixels
self.width = int(1920) # width of the image sensor, pixels


