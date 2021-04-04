import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# grab column titles from txt file
labels = pd.read_csv("labels_structure.txt")
# grab list of filenames of images
files = os.listdir("/mnt/c/Users/gsjns/focus/plant_imaging/images/experiment_20210210_1")
# extract times of each file
times = [s[11:19] for s in files]
# convert timestamps into minutes since experiment start
hh = [int(h[0:2]) for h in times]
hh = np.array(hh)*60
mm = [int(m[3:5]) for m in times]
mm = np.array(mm)
times = hh+mm
start_time = times[0]
times = times-start_time

# not elegant but it works to assign color channel label
channels = []

for x in files: 
    if x[28] == '0':
        channels = np.concatenate((channels,np.array(['G'])))
    elif x[28] == '1':
        channels = np.concatenate((channels,np.array(['B'])))
    elif x[28] == '2':
        channels = np.concatenate((channels,np.array(['NIR'])))
    elif x[28] == '3':
        channels = np.concatenate((channels,np.array(['R'])))



# split into subarrays for each image 
files = np.array(files)

files_cat = np.array_split(files,len(files)/4)
files_cat = [list(i) for i in files_cat]
print(files_cat)
#files_cat = np.asarray(files_cat, dtype=list)
#print(files_cat)
#print(type(files_cat[0]))

# split into subarrays for each image
channels_cat = np.array_split(channels,len(channels)/4)
channels_cat = [list(i) for i in channels_cat]

# cut down original times parsed from filenames
times_cat = times[0::4]

# -1: not stressed yet
# time since stress in minutes
stress_time = "8_57_00" 
stress_time = 8*60+57
stress_time = stress_time - start_time
time_since_stress = times_cat - stress_time

time_since_stress[time_since_stress<0] = -1

# put this data into df
labels.loc[:,"image_name"] = files_cat
labels.loc[:,"channel"] = channels_cat
labels.loc[:,"time"] = times_cat
labels.loc[:,"time_since_stress"] = time_since_stress


print(labels)

labels.to_csv('./images/labels_20210210_01.csv', index=False)
