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
                transform = Mask(8))

    dataloader = torch.utils.data.DataLoader(dataset=dataset, 
            batch_size = 5, 
            shuffle = True)


    print(dataloader[0])

if __name__ == "__main__":
    main()
