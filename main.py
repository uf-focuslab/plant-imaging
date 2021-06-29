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
"""
# MNIST Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 5
num_epochs = 2
learning_rate = 0.01
"""

# PLANT Hyper-parameters
sequence_length = 10 # time list? 
input_size = 256
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 5
num_epochs = 2
learning_rate = 0.01

"""
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
"""

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
        shuffle=True)


"""
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        print(i)
        
        print(images)
        print('RESHAPE::')
        print(images.reshape(-1, sequence_length, input_size))
        images = images.reshape(-1, sequence_length, input_size).to(device)

        labels = labels.to(device)
        break
sys.exit()
"""

                   
"""
# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
"""

"""
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
# input_dim: int, number of channels of input tensor
# hidden_dim: int or list of ints for each layer
# num_layers: int, number of lstm layers stacked 
"""

model = ConvLSTM(input_dim=4, 
                 hidden_dim=64,
                 kernel_size=(3,3), 
                 num_layers=1,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)

model = model.to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
"""
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
"""

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

        labels = stress_times
        labels = labels.to(device, dtype=torch.float32)

        

        
        
        ## I also don't think that the dataloader is going to pull out the labels 
        ## properly straight from the dataset/loader, need to do this manually 
        ## or change the way the dataset is formed. but not sure exactly how the 
        ## mnist dataset returns them. i think manual is the best bet. 
         
        # Forward pass
        outputs, hidden = model(images)
        print('::::OUTPUTS::::')
        print(outputs)
        if type(outputs) == list: outputs = outputs[0]
        
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

sys.exit()
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')


