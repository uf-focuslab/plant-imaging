import sys
sys.path.append("/home/gavinstjohn/plant-imaging")
from dataloader import PlantStressDataset
from dataloader import Mask

import torch
import torchvision


def main(): 


    root = '/home/gavinstjohn/plant-imaging/'

    #dataset = torch.utils.data.DataLoader(
    dataset = PlantStressDataset(
                csv_file = root + 'images/labels_20210210_01.csv',
                img_dir = root + 'images/experiment_20210210_1/',
                transform = Mask(8))

    """ii = 0
    for jj in range(0,4):
        for ii in range(0,339):
            sample = dataset[ii,jj]"""

    output = open("output.txt", "r")
    
    line_list = []
    for line in output:
        try: 
            line_list.append(int(line.strip('\n')))
        except ValueError: 
            print('oops')
    #print(int(line_list[0]))
    print(line_list)
    print(min(line_list))
    print(sum(line_list)/len(line_list))
    print(0.8*min(line_list))


if __name__ == "__main__":
    main()
