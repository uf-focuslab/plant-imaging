import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from ast import literal_eval



root = '/blue/eel6935/gavinstjohn/plant/plant-imaging/'
labels = pd.read_csv(root + 'images/labels_20210210_01.csv')


class PlantStressDataset(Dataset):
    """ plant stress experiment dataset """

    def __init__(self, csv_file, img_dir, transform=None):
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

    def __len__(self):
        # returns length of the labels (how many images)
        return len(self.labels)

    def __getitem__(self, idx):
        # idx is a tuple (int,int) containting two ints
        # idx[0] is the sample in the timeline
        # idx[1] is the experiment 
        #       (0: top left, 1: top right, 2: bottom left, 3: bottom right)
        sample_id = idx[0]
        exp_id = idx[1]
        if torch.is_tensor(sample_id):
            sample_id = sample_id.tolist()

        # grabs image file path (image root directory + image file name)
        img_names = literal_eval(self.labels.iloc[sample_id, 0])
        
        img_name_g = os.path.join(self.img_dir, img_names[0])
        image_g = io.imread(img_name_g)

        img_name_b = os.path.join(self.img_dir, img_names[1])
        image_b = io.imread(img_name_b)

        img_name_nir = os.path.join(self.img_dir, img_names[2])
        image_nir = io.imread(img_name_nir)

        img_name_r = os.path.join(self.img_dir, img_names[3])
        image_r = io.imread(img_name_r)

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

        # pul, 'img_channel': img_channel}
        # pulls in the image
        image = np.array([image_g, image_b, image_nir, image_r])
        # pulls in each of the three labels
        capture_time = self.labels.iloc[sample_id,1]
        stress_time = self.labels.iloc[sample_id,2]
        img_channel = literal_eval(self.labels.iloc[sample_id,3])

        # constructs dict of image and labels
        sample = {'image': image, 'capture_time': capture_time, 'stress_time': stress_time, 'img_channel': img_channel}

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
        images = sample['image']

        # init the mask on the blue layer of the stack 
        mask = images[1] #[1] = blue
        # keep anything less than a B value of blue_threshold (8 is good)
        mask[mask <= self.blue_threshold] = 1
        mask[mask > self.blue_threshold] = 0

        # mask all all the layers
        masked_images = images*mask
        
        # put the (now masked) image stack back into the sample dict
        sample['image'] = masked_images

        # spit back out
        return sample


def main():
    """plant_dataset = PlantStressDataset(
            csv_file='/blue/eel6935/gavinstjohn/plant/plant-imaging/images/labels_20210210_01.csv',
            img_dir='/blue/eel6935/gavinstjohn/plant/plant-imaging/images/experiment_20210210_1/')

    sample_101 = plant_dataset[101,2] # pass a tuple in
    print(sample_101["image"][0])
    print(sample_101["image"][0].shape)"""

    t_plant_dataset = PlantStressDataset(
            csv_file='/blue/eel6935/gavinstjohn/plant/plant-imaging/images/labels_20210210_01.csv',
            img_dir='/blue/eel6935/gavinstjohn/plant/plant-imaging/images/experiment_20210210_1/',
            transform=Mask(8))
    
    t_sample_101 = t_plant_dataset[101,2]
    print(t_sample_101['image'][0])
    print(np.count_nonzero(t_sample_101['image'][0]))

    for ii in 


if __name__ == "__main__": 
    main()
