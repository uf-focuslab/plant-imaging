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
from utilities import net_input

from convlstm import ConvLSTM

from IPython import embed
import sys


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PLANT Hyper-parameters
sequence_length = 10 
input_dim = 4                   # number of image channels
hidden_dim = [128, 64, 8, 1]    # hidden layer dimensions
num_layers = 4                  # number of hidden layers
output_dim = 1                  # size of linear output layer
batch_size = 1                  # number of samples per batch
num_epochs = 100                  # loop over entire dataset this many times
learning_rate = 0.01            # learning rate for gradient descent
kernel_size = (3,3)             # kernel size for convolution layer

# PLANT dataset/loader
p_dataset = PlantStressDataset(
        csv_file='/md0/home/gavinstjohn/plant-imaging/images/labels_20210210_01.csv', # csv file containing labels
        img_dir='/md0/home/gavinstjohn/plant-imaging/images/experiment_20210210_1/',# directory containjng images from experiment
        seq_length=sequence_length, # sequence length for LSTM
        quadrant=0, # which quadrant of the experiment should be subsampled? 
        transform=Mask(8)) # mask the data based on blue value of 8

# dataset split up code:: 
test_split = 0.2 # 20% for test, 80% for train
random_seed = 42 # set the seed s/t it's the same no matter what
shuffle_dataset = True # shuffle dataset? yes please

dataset_size = len(p_dataset) # overall length of dataset
indices = list(range(dataset_size)) # list of indices
split = int(np.floor(test_split * dataset_size)) # split location
if shuffle_dataset:  # yes
    np.random.seed(random_seed) # set the seed
    np.random.shuffle(indices) # shuffle up the indices
train_indices, test_indices = indices[split:], indices[:split] # do the split

train_sampler = SubsetRandomSampler(train_indices) # use subsetrandomsampler from pytorch
test_sampler = SubsetRandomSampler(test_indices) # samples indices randomly 


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


model = ConvLSTM(input_dim=input_dim, # images have 4 channels, R G B NIR
                 hidden_dim=hidden_dim, # hidden layer dimensions, arbitrarily set
                 kernel_size=kernel_size, # size of the kernel for conv layer, also arbitrarily set
                 num_layers=num_layers, # number of hidden layers
                 output_dim=output_dim, # linear output layer dimension
                 batch_first=True,  # not sure what this setting does
                 bias=True, # also unsure
                 return_all_layers=False).to(device) # also unsure


# Loss and optimizer
#criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss() # couldn't get cel criterion to work so tried mse and it ran
#criterion = nn.BCELoss() # for binary classification
# dont think bceloss is working great, would need to do oversampling
# trying crossentropyloss with normalized weights
nSamples = [62, 277] # [unstressed, stressed]
normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
normedWeights = torch.FloatTensor(normedWeights).to(device)
criterion = nn.CrossEntropyLoss(weight=normedWeights) # note: cel needs dtype=long for out/labels

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # yoinked from example, not sure what the best optimizer is


### TRAINING LOOP ###
total_step = len(p_train_loader) # total number of samples in dataset
for epoch in range(num_epochs): # for each epoch, 
    for i, (images,capture_times,stress_times) in enumerate(p_train_loader): # loop over each batch in the dataloader
        # batch size, sequence length, channels, height of image, width of image
        # input.shape = [b, t, c, h, w]
        #               [b, t, 4, 256, 256]
        images = images.reshape(batch_size, sequence_length, 4, 256, 256).to(device, dtype=torch.long)

        #labels = stress_times
        # experiment to try and create binary classifier instead of predicting time since stress
        #labels[labels>0] = 1
        #labels[labels==-1] = 0
        labels = stress_times[:,-1] # only grab the key sample
        labels = labels.reshape(len(labels),1) # rotate to be vertical
        labels[labels>0] = 1 # binary classify
        labels[labels==-1] = 0

        labels = labels.to(device, dtype=torch.long) # sends to cuda device and changes datatype

        # Forward pass
        outputs, hidden, out = model(images) # out is the output of the linear layer, outputs and hidden are spat out by the lstm

        
        sig = nn.Sigmoid() # squish output between 0 and 1
        out = sig(out)
        out = torch.cat((out,1-out),1) # needed for cross entropy loss, shape of (N,C)
        loss = criterion(out, labels[0])
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        print(i)
        optimizer.step()
        
        # little status updates
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Save the model checkpoint
torch.save(model.state_dict(), 'cel_model_100.ckpt')

print('TESTING START')
# model testing is not working, need to yoink a working one
# Test the model
model.eval()
with torch.no_grad():
    #correct = 0
    #total = 0
    
    test_loss = 0
    accuracy = 0

    for images,capture_times,stress_times in p_test_loader:

        # pretty much a copy of the forward loop
        images = images.reshape(batch_size, sequence_length, 4, 256, 256).to(device, dtype=torch.long)

        """
        labels = stress_times
        labels[labels>0] = 1
        labels[labels==-1] = 0
        """
        labels = stress_times[:,-1] # only grab the key sample
        labels = labels.reshape(len(labels),1) # rotate to be vertical
        labels[labels>0] = 1 # binary classify
        labels[labels==-1] = 0
        print('labels: ', labels)

        labels = labels.to(device, dtype=torch.long)


        outputs, hidden, out = model(images)
        out = sig(out)
        out = torch.cat((out,1-out),1) # needed for cross entropy loss, shape of (N,C)
        print('out: ', out)

        #loss = criterion(out, labels[0])
        test_loss += criterion(out, labels[0]).item()

        ps = torch.exp(out)
        #equality = (labels.data == ps.max(dim=1)[1])
        equality = (labels.data == ps)
        accuracy += equality.type(torch.FloatTensor).mean()

    print('test_loss: ', test_loss)
    print('accuracy: ', accuracy)
    #print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total)) 



