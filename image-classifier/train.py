# Imports here

import torch
from torch import nn
from torch import optim
from torchvision import models
import argparse

from data_functions import get_data, get_loader
from ml_functions import train, test

# Set arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--save_dir', default='')
parser.add_argument('--arch', default='vgg19')
parser.add_argument('--epochs', default=5)
parser.add_argument('--hidden_units', default=5000)
parser.add_argument('--learnrate', default=0.003)
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()

# Load data and create data loaders
train_data, valid_data, test_data = get_data(args.data_dir)
trainloader, validloader, testloader = get_loader(train_data, valid_data, test_data)
 
# Load model
if args.arch =='vgg19':
    model = models.vgg19(pretrained=True)

if args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)

# Define hyperparameters
learn_rate = float(args.learnrate)
epochs = int(args.epochs)
fc1_out = fc2_in = int(args.hidden_units)
fc2_out = output_in = 1000

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Define new classifier to be trained on flower datset with 102 classes
model.classifier = nn.Sequential(nn.Linear(25088, fc1_out),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(fc2_in, fc2_out),
                                 nn.ReLU(),
                                 nn.Linear(output_in, 102),
                                 nn.LogSoftmax(dim=1))

model.criterion = criterion=nn.NLLLoss()
model.optimizer = optimizer=optim.Adam(model.classifier.parameters())

# Use GPU if it's available
if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
    
model.to(device)

# Train model
model = train(model, trainloader, validloader, device, epochs=epochs)

# Test accuracy
test(model, testloader, device)

# Save checkpoint
checkpoint = {'state_dict': model.state_dict(),
             'class_to_idx': train_data.class_to_idx, 
             'hidden_units': args.hidden_units,
             'arch': args.arch}

if args.save_dir:
    save_path = save_dir + '\checkpoint.pth'
else:
    save_path = 'checkpoint.pth'

torch.save(checkpoint, save_path)

print('Checkpoint saved.')


    


