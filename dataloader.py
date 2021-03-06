import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from ast import literal_eval
import cv2



root = '/home/gavinstjohn/plant-imaging/'
labels = pd.read_csv(root + 'images/labels_20210210_01.csv')


class PlantStressDataset(Dataset):
    """ plant stress experiment dataset """

    def __init__(self, csv_file, img_dir, seq_length, seq_range, quadrant, transform=None):
        """
        args: 
            csv_file [str]: path to csv file with labels
            img_dir  [str]: path to directory with all the images
            transform [callable, optional]: optional transform to 
                be applied on a sample
        """
        
        # grab info from the init args
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.seq_length = seq_length
        self.seq_range = seq_range
        self.quadrant = quadrant

    def __len__(self):
        # returns length of the labels (how many images)
        return len(self.labels)

    def __grab__(self, sample_id):
        exp_id = self.quadrant

        # grabs image file path (image root directory + image file name)
        img_names = literal_eval(self.labels.iloc[sample_id, 0])
        
        img_name_g = os.path.join(self.img_dir, img_names[0])
        image_g = cv2.imread(img_name_g, 0)

        img_name_b = os.path.join(self.img_dir, img_names[1])
        image_b = cv2.imread(img_name_b, 0)

        img_name_nir = os.path.join(self.img_dir, img_names[2])
        image_nir = cv2.imread(img_name_nir, 0)

        img_name_r = os.path.join(self.img_dir, img_names[3])
        image_r = cv2.imread(img_name_r, 0)

        img_names = [img_name_g, img_name_b, img_name_nir, img_name_r]


        # cut up images on quadrant basis
        # i feel like there's a more concise way to do this
        # also might be worth changing the cases to match the quadrants of an x-y plane 
        # ie I is top right, II top left, III bottom left, IV bottom right
        h,w = image_g.shape # grab size of img, could use any of them 
        height_cutoff = h//2 # get the middles of height and width
        width_cutoff = w//2
        # based on which experiment id is passed in return a different quadrant

        if exp_id == 0:
            image_g = image_g[:height_cutoff, :width_cutoff]
            image_b = image_b[:height_cutoff, :width_cutoff]
            image_nir = image_nir[:height_cutoff, :width_cutoff]
            image_r = image_r[:height_cutoff, :width_cutoff]
        elif exp_id == 1:
            image_g = image_g[:height_cutoff, width_cutoff:]
            image_b = image_b[:height_cutoff, width_cutoff:]
            image_nir = image_nir[:height_cutoff, width_cutoff:]
            image_r = image_r[:height_cutoff, width_cutoff:]
        elif exp_id == 2:
            image_g = image_g[height_cutoff:, :width_cutoff]
            image_b = image_b[height_cutoff:, :width_cutoff]
            image_nir = image_nir[height_cutoff:, :width_cutoff]
            image_r = image_r[height_cutoff:, :width_cutoff]
        elif exp_id == 3:
            image_g = image_g[height_cutoff:, width_cutoff:]
            image_b = image_b[height_cutoff:, width_cutoff:]
            image_nir = image_nir[height_cutoff:, width_cutoff:]
            image_r = image_r[height_cutoff:, width_cutoff:]

        # pulls in the image
        image = np.array([image_g, image_b, image_nir, image_r])

        return image

    def __getitem__(self, idx):
        # idx is an int referring to the sample value
        # quadrant is now controlled through the quadrant var in __init__
        sample_id = idx
        exp_id = self.quadrant
        seq_length = self.seq_length
        seq_range = self.seq_range
        
        if (seq_range%seq_length) != 0: raise Exception('seq_range: {} must be evenly divisible by seq_length: {}'.format(seq_range,seq_length))


        """
        # handle the edges of the dataset. there must be a more elegant way to do this
        if sample_id-seq_length<0: sample_id = sample_id+seq_length
        """
        
        """
        # this will make an array which goes from t-seq_range to t, spaced at seq_range/seq_length
        # id: [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100] for t = 100, range = 40, length = 10
        time_series = torch.arange(sample_id-(seq_range-(seq_range/seq_length)), sample_id+(seq_range/seq_length), seq_range/seq_length, dtype=int)


        # a bit hacky to implement adjustable time series
        # if the time series predates the start of the experiment, shift it forward
        start = torch.clone(time_series[0])
        if time_series[0] < 0: time_series-=start
        """

        if torch.is_tensor(sample_id):
            batch_list = sample_id.tolist()

            # for batches, this will generate [b,t,c,h,w]
            image = np.zeros((len(batch_list),seq_length,4,256,256))
            capture_time = stress_time = np.zeros((len(batch_list),seq_length))
            for j, super_sample in enumerate(batch_list): 

                # this will make an array which goes from t-seq_range to t, spaced at seq_range/seq_length
                # id: [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100] for t = 100, range = 40, length = 10
                time_series = torch.arange(super_sample-(seq_range-(seq_range/seq_length)), super_sample+(seq_range/seq_length), seq_range/seq_length, dtype=int)

                # a bit hacky to implement adjustable time series
                # if the time series predates the start of the experiment, shift it forward
                start = torch.clone(time_series[0])
                if time_series[0] < 0: time_series-=start

                #for i, sample in enumerate(range(super_sample-seq_length, super_sample)):
                for i, sample in enumerate(time_series.tolist()):
                    # pulls in each of the three labels
                    capture_time[j,i] = self.labels.iloc[sample,1]
                    stress_time[j,i] = self.labels.iloc[sample,2]
                    
                    image[j,i] = self.__grab__(sample)

        elif type(sample_id)==int: 
            # pulls in each of the three labels
            capture_time = self.labels.iloc[sample_id,1]
            stress_time = self.labels.iloc[sample_id,2]

            # for single pulls, this will generate [t,c,h,w]
            image = np.zeros((seq_length,4,256,256))
            # and labels will be [t, label_value]
            capture_time = stress_time = np.zeros((seq_length))
                
            # this will make an array which goes from t-seq_range to t, spaced at seq_range/seq_length
            # id: [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100] for t = 100, range = 40, length = 10
            time_series = torch.arange(sample_id-(seq_range-(seq_range/seq_length)), sample_id+(seq_range/seq_length), seq_range/seq_length, dtype=int)

            # a bit hacky to implement adjustable time series
            # if the time series predates the start of the experiment, shift it forward
            start = torch.clone(time_series[0])
            if time_series[0] < 0: time_series-=start

            #for i, sample in enumerate(range(sample_id-seq_length, sample_id)):
            for i, sample in enumerate(time_series.tolist()):
                capture_time[i] = self.labels.iloc[sample,1]
                stress_time[i] = self.labels.iloc[sample,2]
            
                image[i] = self.__grab__(sample)


        # constructs dict of image and labels
        #sample = {'image': image, 'capture_time': capture_time, 'stress_time': stress_time}
        # 6/22: changing this to a list to play nicer with pytorch
        sample = [image, capture_time, stress_time, time_series]


        if self.transform:
            sample = self.transform(sample)

        return sample


class Mask(object):
    """
    Mask each image in a sample based on the blue pixel threshold 
        and control quadrant

    Args: 
        blue_threshold [int, 0-255]: threshold of blue pixel value to be masked 
            (8 works well)

        NOT BEING USED::
        control_mask [2d binary tensor]: binary mask of the control quadrant
            must be same size as images

    """

    def __init__(self, blue_threshold):
        # make sure the blue_threshold is the correct datatype
        assert isinstance(blue_threshold, int)
        # set the internal variable to the passed in blue_threshold
        self.blue_threshold = blue_threshold

    def __call__(self, sample):
        # pull the image stack out of the sample dict
        images = sample[0]

        # init the mask on the blue layer of the stack 
        mask = images[0][1] #[1] = blue
        # keep anything less than a B value of blue_threshold (8 is good)
        mask[mask <= self.blue_threshold] = 1
        mask[mask > self.blue_threshold] = 0


        # mask all all the layers
        masked_images = images*mask
        
        # put the (now masked) image stack back into the sample dict
        sample[0] = masked_images

        # spit back out
        return sample


def main():
    """plant_dataset = PlantStressDataset(
            csv_file='/md0/home/gavinstjohn/plant-imaging/plant-imaging/images/labels_20210210_01.csv',
            img_dir='/md0/home/gavinstjohn/plant-imaging/plant-imaging/images/experiment_20210210_1/')

    sample_101 = plant_dataset[101,2] # pass a tuple in
    # sample rules are: 0: image, 1: capture time, 2: stress time
    print(sample_101[0][0])
    print(sample_101[0][0].shape)"""

    t_plant_dataset = PlantStressDataset(
            csv_file='/md0/home/gavinstjohn/plant-imaging/images/labels_20210210_01.csv',
            img_dir='/md0/home/gavinstjohn/plant-imaging/images/experiment_20210210_1/',
            transform=Mask(8))
    



    sample = t_plant_dataset[101,0][0][0]

    mask = sample
    mask[mask > 0] = 1

    # elementwise multiply unraveled mask           [ 1 1 0 0 1 1 0 0 0 ... 0 0 0 ]
    # with arange array (all indices of mask array) [ 0 1 2 3 4 5 6 7 8 ... 65536 ]
    # which results in an array of indices where the masked values are zero'd
    #                                               [ 0 1 0 0 4 5 0 0 0 ... 0 ]
    # remove all zeros                              [ 1 4 5 ... ]
    # add back in the leading index if mask[0] is 1 [ 0 1 4 5 ... ]
    # take random sample of this array to get 15000 random elements which are not zero
    
    # elementwise multiply
    B = np.multiply(np.ravel(mask),np.arange(mask.size))
    # remove all zeros
    B = B[B!=0]
    # add back in leading zero
    if np.ravel(mask)[0] == 1: append(0,B)
    # random sample of indices
    pixel_locations = np.random.choice( B, replace=False, size=15000)
    # unravel mask, everything is pretty much going to be worked on unraveled from 
    # here forward
    mask = np.ravel(mask)
    # zero out all locations in mask which are not in the random sample
    mask[~pixel_locations] = 0


    # construct all the time spots, 
    # in this instance creates [95,96,97,...,104,105] (100 +/-5)
    time = 100
    quadrant = 0
    time_range = np.arange(time-5,time+6)

    # initialize 
    net_input = np.zeros((4,len(time_range),256**2))
    # fill in network input matrix
    for t, time in enumerate(time_range): 
        # for each timestep in the time_range [95,96,97,...,103,104,105]
        # and for each color channel in the image
        for channel in range(4): 
            # grab the sample at that timestep and the specified quadrant
            temp_sample = t_plant_dataset[time,quadrant][0][channel]
            # mask it to the selected random pixel_locations
            masked_sample = np.multiply(np.ravel(temp_sample),mask)
            # slot it into big input matrix
            net_input[channel,t,:] = masked_sample

    print(net_input)

if __name__ == "__main__": 
    main()
