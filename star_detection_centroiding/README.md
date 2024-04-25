# Deep Learning Methods for Star Detection and Centroiding
### Author: Hongrui Zhao hongrui5@illinois
### Date: Sep/19/2023

## (1) `neural_net` folder 
It contains different neural networks. Currently, we have original UNet (named as `CentroidNet` here) and MobileUnet.

## (2) Root directory 
`visualization.py`: validate a selected trained model with `test` dataset. Visualize prediction output of ground truth vs deep learning method vs the conventional threshold method.
`evaluation.py`: evaluate deep learning method and the conventional threshold method. 

## (3) Notes for `training.py`
To avoid GPU running out of memory, a single image batch size is selected.

## (4) `onnx_conversion.py`
Convert neural networks saved in `ACES\star_detection_centroiding\saved_models` to onnx file.

The output onnx files will be saved under `ACES\star_detection_centroiding`.