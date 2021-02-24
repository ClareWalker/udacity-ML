# Imports here

import torch
from torch import nn
from torch import optim
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd

from workspace_utils import keep_awake

# Functions here

def train(model, trainloader, validloader, device, epochs=5):
    # Train fully connected layers
    print_every = 25
    steps = 0
    optimizer = model.optimizer
    criterion = model.criterion

    for e in keep_awake(range(epochs)):
        running_loss = 0
        # turn on dropout for training
        model.train()
        for images, labels in trainloader:
            steps +=1
        
            # move images and labels to same device as model
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            # print progress periodically
            if steps % print_every == 0:
                # turn off dropout for validation
                model.eval()
            
                accuracy = 0
                valid_loss = 0
            
                with torch.no_grad():
                    for images, labels in validloader:
                        # move images and labels to same device as model
                        images, labels = images.to(device), labels.to(device)
                
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                
                        valid_loss += batch_loss.item()
                
                        # check accuraccy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
                # print output
                print(f"Epoch {e+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")
    
    return model

def test(model, testloader, device):
    model.eval()
            
    accuracy = 0
    test_loss = 0
    
    criterion = model.criterion
            
    with torch.no_grad():
        for images, labels in testloader:
            # move images and labels to same device as model
            images, labels = images.to(device), labels.to(device)
                
            logps = model.forward(images)
            batch_loss = criterion(logps, labels)
                
            test_loss += batch_loss
                
            # check accuraccy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    # print output
    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
         f"Test accuracy: {accuracy/len(testloader):.3f}")
    
def load_checkpoint(filepath,fc2_out=1000, output_in=1000):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(filepath, map_location=map_location)
    
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    fc1_out = fc2_in = hidden_units
    
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    
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


    # Load weights, biases and class to indices
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def predict(image_path, model, device, top_k=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    loader = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    image = Image.open(image_path)
    image = loader(image).float()
    image = image.unsqueeze(0)
    image = image.to(device)
    
    ps = torch.exp(model(image))
    
    ps = ps.to('cpu')
    probs, indices = ps.topk(top_k, dim=1)
    probs = probs[0].detach().numpy()
    
    idx_to_class = inv_map = {v: k for k, v in model.class_to_idx.items()}    
    
    classes = [idx_to_class[i] for i in indices[0].tolist()]

    return probs, classes 