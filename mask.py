import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


image_directory = "/mnt/c/Users/gsjns/focus/plant_imaging/images/experiment_20210210_1/"
output_directory = "/mnt/c/Users/gsjns/focus/plant_imaging/images/output/"

#   date: yyyy_mm_dd
experiment_date = "2021_02_10_"

#   time: hh_mm_ss
#  first: 07_55_02
# stress: 08_57_56
#   last: 13_38_22 
frame = "07_55_02"

# construct paths for each channel
nir_image = image_directory + experiment_date + frame + "multi_0CH2.tiff"
r_image = image_directory + experiment_date + frame + "multi_0CH3.tiff"
b_image = image_directory + experiment_date + frame + "multi_0CH1.tiff"
g_image = image_directory + experiment_date + frame + "multi_0CH0.tiff"

# load each channel into opencv
nir = cv2.imread(nir_image, cv2.IMREAD_GRAYSCALE)
red = cv2.imread(r_image, cv2.IMREAD_GRAYSCALE)
blue = cv2.imread(b_image, cv2.IMREAD_GRAYSCALE) 
green = cv2.imread(g_image, cv2.IMREAD_GRAYSCALE)

# NDVI implementation
# doesn't work well

# ndvi = (blue - red)/(blue + red)
# mask = ndvi
# mask[mask <= 0] = 0
# mask[mask > 0] = 1

# blue filter implementation
# works ok

mask = blue
# filter pixels which have a blue value of < 8
mask[mask <= 8] = 1
mask[mask > 8] = 0

# NOTE remove
# only here for eye-visibility
mask = mask*255

# save visible mask
# cv2.imwrite(output_directory + "blue_mask_frame" + frame + ".png", mask)

# build and save an rgb reconstruction of the frame
r = Image.fromarray(red)
g = Image.fromarray(green)
b = Image.fromarray(blue)
rgb = Image.merge('RGB',(r,g,b))
rgb.save(output_directory + "rgb_frame" + frame + ".png")
