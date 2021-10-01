import sys
sys.path.append("/md0/home/gavinstjohn/plant-imaging")
from dataloader import PlantStressDataset
from dataloader import Mask

import torch
import torchvision


def main(): 


    root = '/md0/home/gavinstjohn/plant-imaging/'

    #dataset = torch.utils.data.DataLoader(
    dataset = PlantStressDataset(
                csv_file = root + 'images/labels_20210210_01.csv',
                img_dir = root + 'images/experiment_20210210_1/',
                seq_length = 10, 
                seq_range = 80,
                quadrant = 0,
                transform = Mask(8))
                

    dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            batch_size = 1, 
            shuffle = False)

    for i, (images, capture_times, stress_times, time_series) in enumerate(dataloader): 
        #print(i, time_series)
        print(images.size())
        break





if __name__ == "__main__":
    main()
