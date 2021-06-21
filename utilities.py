import numpy as np
from dataloader import PlantStressDataset
from dataloader import Mask

def net_input(time_list, quadrant, N, experiment_info):
    """ input: 
        time_list [array of ints]: [95, 96, ... , 100, ..., 104, 105]
        quadrant [int]: 0-3 0: top left, 1: top right, 2: bottom left, 3: bottom right
        N [int]: number of pixel locations to be sampled from the image, 15000 baseline
        experiment_info [list]: ['label csv file path', 'experiment image directory'] 

        output: 
        net_input [np matrix]: 4 channels x # of timesteps x N pixel locations
    """
    
    dataset = PlantStressDataset(
            csv_file = experiment_info[0], 
            img_dir = experiment_info[1], 
            transform = Mask(8))

    # reconstruct the binary mask of foliage
    mask = dataset[time_list[0],quadrant]['image'][0]
    mask[mask > 0] = 1

    # elementwise multiply unraveled mask           [ 1 1 0 0 1 1 0 0 0 ... 0 0 0 ]
    # with arange array (all indices of mask array) [ 0 1 2 3 4 5 6 7 8 ... 65536 ]
    # which results in an array of indices where the masked values are zero'd
    #                                               [ 0 1 0 0 4 5 0 0 0 ... 0]
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
    pixel_locations = np.random.choice(B, replace=False, size=N)
    # unravel mask, everything is pretty much going to be worked on unraveled from here forward
    mask = np.ravel(mask)
    # zero out all locations in mask which are not in the random sample
    mask[~pixel_locations] = 0


    # initialize
    net_input = np.zeros((4,len(time_list),(mask.shape[0])**2))
    # fill in network input matrix
    for t, time in enumerate(time_list):
        # for each timestep in the time_list [95,96,97,...,103,104,105]
        # and for each color channel in the image
        for channel in range(4):
            # grab the sample at that timestep and the specified quadrant
            temp_sample = dataset[time,quadrant]['image'][channel]
            # mask it to the selected random pixel_locations
            masked_sample = np.multiply(np.ravel(temp_sample),mask)
            # slot it into big input matrix
            net_input[channel,t,:] = masked_sample

    return net_input
