import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from dataloader import PlantStressDataset
from dataloader import Mask
from utilities import net_input

from convlstm import ConvLSTM

from IPython import embed
import sys


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PLANT Hyper-parameters
sequence_length = 10 # time list? 
input_size = 256
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 2
num_epochs = 1
learning_rate = 0.01

# PLANT dataset/loader

p_train_dataset = PlantStressDataset(
        csv_file='/md0/home/gavinstjohn/plant-imaging/images/labels_20210210_01.csv',
        img_dir='/md0/home/gavinstjohn/plant-imaging/images/experiment_20210210_1/',
        seq_length=sequence_length,
        quadrant=0,
        transform=Mask(8))

p_train_loader = torch.utils.data.DataLoader(
        dataset=p_train_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True)


model = ConvLSTM(input_dim=4, 
                 hidden_dim=[128, 64, 8, 1],
                 kernel_size=(3,3), 
                 num_layers=4,
                 output_dim=10,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False).to(device)


# Loss and optimizer
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


### UNDER CONSTRUCTION ####
total_step = len(p_train_loader)
for epoch in range(num_epochs):
    for i, (images,capture_times,stress_times) in enumerate(p_train_loader):
        ## image reshaping happens here, may be where to use net_input function 
        ## i think it grabs all the images for the batch size, not sure how that 
        ## fits into the weird network input shape we're using. 

        #images = images.reshape(-1, sequence_length, input_size).to(device, dtype=torch.float32)
        #put = images.reshape(batch_size, sequence_length 
        #images = images.reshape(batch_size, sequence_length, images
        #images = images.reshape(batch_size, -1, input_size).to(device, dtype=torch.float32)

        # input.shape = [b, t, c, h, w]
        #               [b, t, 4, 256, 256]
        #               [5, 10, 4, 256, 256]
        images = images.reshape(batch_size, sequence_length, 4, 256, 256).to(device, dtype=torch.float32)

        labels = stress_times.view(batch_size*sequence_length)
        labels = stress_times
        labels[labels>0] = 1
        labels[labels==-1] = 0

        labels = labels.to(device, dtype=torch.float32)

        ## I also don't think that the dataloader is going to pull out the labels 
        ## properly straight from the dataset/loader, need to do this manually 
        ## or change the way the dataset is formed. but not sure exactly how the 
        ## mnist dataset returns them. i think manual is the best bet. 
         
        # Forward pass
        outputs, hidden, out = model(images)
        if type(outputs) == list: outputs = outputs[0]
        if type(hidden) == list: hidden = hidden[0][0]
        
        loss = criterion(out, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        print(i)
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images,capture_times,stress_times in p_train_loader:


        images = images.reshape(batch_size, sequence_length, 4, 256, 256).to(device, dtype=torch.float32)

        labels = stress_times.view(batch_size*sequence_length)
        labels = stress_times
        labels[labels>0] = 1
        labels[labels==-1] = 0

        labels = labels.to(device, dtype=torch.float32)


        outputs, hidden, out = model(images)
        #_, predicted = torch.max(outputs.data, 1)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')


