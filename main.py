import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# disable later for speed, used to debug 'cuda error: device side assert'


import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from dataloader import PlantStressDataset
from dataloader import Mask
from dataloader import polarize_plant
from utilities import net_input

from convlstm import ConvLSTM
from cnn_simple import CNN

from IPython import embed
import argparse
import sys


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PLANT Hyper-parameters
sequence_length = 4 
sequence_range = 40
input_dim = 4                   # number of image channels
num_layers = 4                  # number of hidden layers
output_dim = 1                  # size of linear output layer
batch_size = 1                  # number of samples per batch // batch size >1 broken right now
num_epochs = 10                 # loop over entire dataset this many times
learning_rate = 0.0001            # learning rate for gradient descent
kernel_size = 4                 # kernel size for convolution layer

# cmd arguments
parser = argparse.ArgumentParser()

# -te MODEL_FILE -tr MODEL_FILE
parser.add_argument("-te", "--test", dest="test_file", default=False, help="filename of model file to be tested")
parser.add_argument("-tr", "--train", dest="train_file", default=False,  help="filename of model file to be written to after training")

args = parser.parse_args()
train_file, test_file = args.train_file, args.test_file

print("train: {}, test: {}".format(train_file, test_file))

# possible permutations: 
# neither specified: runs through training and testing without saving model file
# train specified: runs through training and saves to specified file before testing 
# test specified: does not train, imports test file and tests
# both specified (must be same file): runs through training and saves to specified file before testing specified file, same as case 1
if (type(train_file)==str and type(test_file)==str) and test_file!=train_file: 
    raise Exception("Both train and test files are specifed but not the same file")

# experimenting whether network can pick up on extremely simplified images, polarize_plant makes unstressed images black, stressed stay as plants. 
multi_transform = transforms.Compose([
        Mask(8), 
        polarize_plant,])

# PLANT dataset/loader
p_dataset = PlantStressDataset(
        csv_file='/md0/home/gavinstjohn/plant-imaging/images/labels_20210210_01.csv', # csv file containing labels
        img_dir='/md0/home/gavinstjohn/plant-imaging/images/experiment_20210210_1/',# directory containjng images from experiment
        seq_length=sequence_length, # sequence length for LSTM
        seq_range=sequence_range,
        quadrant=0, # which quadrant of the experiment should be subsampled? 
        transform=polarize_plant(True)) # mask the data based on blue value of 8

# annoying problem: when constructing time_series @ low indices it tries to pull negative time values. 
# initially the bandaid fix was when it attempted that to push the value forward to t=0, 
# that resulted in oversampling time_series which originated @ t=0
# so new fix: create a subset which is pushed forward, then make that the target of the enumerate iterator
# after inside the loop, sample from original (unsubsampled dataset) s/t the time_series can still be constructed

push_value = sequence_range-(sequence_range/sequence_length)
if not (int(push_value) == push_value): 
    raise Exception('sequence_range and sequence_length aren\'t lining up to create int indices, check them')

push_indices = range(int(push_value), len(p_dataset))
push_p_dataset = torch.utils.data.Subset(p_dataset,push_indices) 

# dataset split up code:: 
test_split = 0.2 # 20% for test, 80% for train
random_seed = 42 # set the seed s/t it's the same no matter what
shuffle_dataset = False # shuffle dataset? yes please

# set the manual seed
torch.manual_seed(random_seed)

# this needs to be overhauled s/t stressed and unstressed samples are equally represented in test/train
# generate two sets of dataloaders, one to sample from (dataset) and one to point the enumerator at (push_dataset)
"""
dataset_size = len(p_dataset)
indices = list(range(dataset_size)) # list of indices
split = int(np.floor(test_split * dataset_size)) # split location
"""

push_dataset_size = len(push_p_dataset) # overall length of dataset
push_indices = list(range(push_dataset_size)) # list of indices
push_split = int(np.floor(test_split * push_dataset_size)) # split location
if shuffle_dataset:  # yes
    np.random.seed(random_seed) # set the seed 
#    np.random.shuffle(indices) # shuffle up the indices
    np.random.shuffle(push_indices)

# setting up sample weights for weightedRandomSampler
# goal: train dataset has equal # of stressed and unstressed samples
# count # of stressed / unstressed samples
nSamples = [0, 0] # [unstressed, stressed] 
target = torch.zeros(len(push_p_dataset))
for i in range(len(push_p_dataset)):
    _,_,_,stress_times,_ =push_p_dataset[i]
    label = stress_times[-1]
    if label>0:     
        nSamples[1] += 1
        target[i] = 1
    elif label==-1: 
        nSamples[0] += 1
        target[i] = 0


weight = 1. / torch.Tensor(nSamples)
push_indices = torch.Tensor(push_indices).long()
samples_weight = torch.Tensor([weight[int(t)] for t in target[push_indices]])

#train_indices, test_indices = indices[split:], indices[:split] # do the split
push_train_indices, push_test_indices = push_indices[push_split:], push_indices[:push_split] # do the split

"""
train_sampler = SubsetRandomSampler(train_indices) # use subsetrandomsampler from pytorch
test_sampler = SubsetRandomSampler(test_indices) # samples indices randomly 
"""

push_train_sampler = SubsetRandomSampler(push_train_indices) # use subsetrandomsampler from pytorch
push_test_sampler = SubsetRandomSampler(push_test_indices) # samples indices randomly 

# use push_ dataloaders to generate which samples to grab, but actually grab them directly from the p_dataset
"""
p_train_loader = torch.utils.data.DataLoader(
        dataset=p_dataset, # choose dataset
        batch_size=batch_size, # set batch size
        drop_last=True, # drop final batch, it often is not divisible by the batch size and breaks stuff
        sampler=train_sampler) # sample from the train sample
p_test_loader = torch.utils.data.DataLoader(
        dataset=p_dataset, # choose dataset
        batch_size=batch_size, # set batch size
        drop_last=True, # drop final batch, it often is not divisible by the batch size and breaks stuff
        sampler=test_sampler) # sample from the test sample
        """

push_p_train_loader = torch.utils.data.DataLoader(
        dataset=push_p_dataset, # choose dataset
        batch_size=batch_size, # set batch size
        drop_last=True, # drop final batch, it often is not divisible by the batch size and breaks stuff
        sampler=push_train_sampler) # sample from the train sample
push_p_test_loader = torch.utils.data.DataLoader(
        dataset=push_p_dataset, # choose dataset
        batch_size=batch_size, # set batch size
        drop_last=True, # drop final batch, it often is not divisible by the batch size and breaks stuff
        sampler=push_test_sampler) # sample from the test sample

# CNN model 
model = CNN(input_dim=input_dim, kernel_size=kernel_size).to(device)


# Loss and optimizer
# trying crossentropyloss with normalized eights
# this needs to be automatically set eventually
# push value pushes unstressed values out
nSamples = [62-push_value, 277] # [unstressed, stressed]
normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
print(normedWeights)
# test manually set weights
#normedWeights = [0.1, 0.9]
normedWeights = torch.FloatTensor(normedWeights).to(device)
criterion = nn.CrossEntropyLoss(weight=normedWeights) # note: cel needs dtype=long for out/labels

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # yoinked from example, not sure what the best optimizer is
# changed to SGD from adam just for a test, adam is supposed to have much faster convergence
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # yoinked from example, not sure what the best optimizer is


# only activate training loop if a destination model training file is specified
if train_file: 
    ### TRAINING LOOP ###
    total_step = len(push_p_train_loader) # total number of samples in dataset
    loss_summary = []
    for epoch in range(num_epochs): # for each epoch, 
        for i, (sample_id,_,_,_,_) in enumerate(push_p_train_loader): # loop over each batch in the dataloader
            # skip early values based on hyperparameters s/t unstressed samples are not oversampled
            # if the time_series ends at a value which is too low s/t the beginning of the stime_series predates
            # the start of the experiment, skip that iteration.
            # this will cause some problems with the test/val split 
            # need to adjust size of dataset. 
            # regrab the values out of the main dataset with an i value pushed from the enumerator
            _, images, _, stress_times, time_series = p_dataset[int(sample_id)]
            images = torch.Tensor(images)
            stress_times = torch.Tensor(stress_times)

            #time_series = torch.Tensor(time_series)
            # this should NOT pop anymore
            if time_series[-1] < push_value:
                raise Exception('it wants samples predating experiment start')

            # batch size, sequence length, channels, height of image, width of image
            # input.shape = [b, t, c, h, w]
            #               [b, t, 4, 256, 256]
            images = images.reshape(batch_size, sequence_length, 4, 256, 256).to(device, dtype=torch.long)

            #labels = stress_times
            # experiment to try and create binary classifier instead of predicting time since stress
            #labels[labels>0] = 1
            #labels[labels==-1] = 0
            labels = stress_times[-1] # only grab the key sample
            #labels = labels.reshape(len(labels),1) # rotate to be vertical ### YO FUTURE GAVIN FIXING THE BATCHSIZZE THING
            labels[labels>0] = 1 # binary classify
            labels[labels==-1] = 0

            labels = torch.Tensor([labels.item()])
            labels = labels.to(device, dtype=torch.long) # sends to cuda device and changes datatype

            # Forward pass
            #_, _, out = model(images) # out is the output of the linear layer, outputs and hidden are spat out by the lstm
            out = model(images)

            
            out = torch.cat((out,1-out),1) # needed for cross entropy loss, shape of (N,C)
            loss = criterion(out, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            print(i)
            optimizer.step()
            loss_summary.append(loss.item())
            
            # little status updates
            if (i+1) % 10 == 0:
                loss_summary = np.mean(np.array(loss_summary))
                print ('Epoch [{}/{}], Step [{}/{}], Loss over 10 steps: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss_summary.item()))
                loss_summary = []


# Save the model checkpoint if a train_file is specified
if type(train_file)==str: 
    torch.save(model.state_dict(), train_file)

# activate no matter what: if train file is specified -> on that file
# if train file is not specified -> on test file
# if neither are specified -> on one-off model
if type(test_file)==str: 
    #torch.load('/md0/home/gavinstjohn/plant-imaging/' + test_file)
    model.load_state_dict(torch.load('/md0/home/gavinstjohn/plant-imaging/' + test_file, map_location="cuda:0"))
    model.to(device)
    print('MODEL LOADED')
    print(model)
print('TESTING START')
# model testing is not working, need to yoink a working one
# Test the model
model.eval()
with torch.no_grad():
    #correct = 0
    #total = 0
    
    test_loss = 0
    accuracy = 0
    correct_stressed = 0
    total_stressed = 0
    correct_unstressed = 0
    total_unstressed = 0

    for i, (sample_id,_,_,_,_) in enumerate(push_p_test_loader):

        _, images, _, stress_times, time_series = p_dataset[int(sample_id)]

        images = torch.Tensor(images)
        stress_times = torch.Tensor(stress_times)
        #time_series = torch.Tensor(time_series)
        # pretty much a copy of the forward loop
        images = images.reshape(batch_size, sequence_length, 4, 256, 256).to(device, dtype=torch.long)

        """
        labels = stress_times
        labels[labels>0] = 1
        labels[labels==-1] = 0
        """
        labels = stress_times[-1] # only grab the key sample
        #labels = labels.reshape(len(labels),1) # rotate to be vertical ### YO FUTURE GAVIN FIXING THE BATCH SIZE THING
        labels[labels>-1] = 1 # binary classify
        labels[labels==-1] = 0

        labels = torch.Tensor([labels.item()])
        labels = labels.to(device, dtype=torch.long)

        out = model(images)
        out = torch.cat((out,1-out),1) # needed for cross entropy loss, shape of (N,C)

        print('###')
        print('i: ', i)
        print("stress_times: ", stress_times)
        print("time_series: ", time_series)
        print('target: ', labels.item(), ' vs: ', out[0])
        


        #loss = criterion(out, labels)
        test_loss += criterion(out, labels)
        equality = (labels.data == out.argmax())
        if not equality: 
            print('WRONG')
        else: 
            print('RIGHT')
        accuracy += equality.type(torch.FloatTensor).mean()
        print('###')
        print('')

        # track distribution of correct and total values
        if out.argmax()==0 and equality: correct_unstressed+=1
        elif out.argmax()==1 and equality: correct_stressed+=1

        if labels[0]==0: total_unstressed+=1
        elif labels[0]==1: total_stressed+=1


    print('test_loss: ', test_loss)
    print('accuracy: {}/{}'.format(int(accuracy),len(push_p_test_loader)))
    print('correct stress predictions: {}/{}'.format(correct_stressed,total_stressed))
    print('correct unstressed predictions: {}/{}'.format(correct_unstressed,total_unstressed))
    #print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total)) 



