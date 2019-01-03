import argparse
import time
import datetime
import os
from collections import OrderedDict

    
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser()

# required arguments

parser.add_argument("data_directory", help="Data directory of the data", type=str)

# optional arguments

parser.add_argument("--save_dir", help="Directory to save the checkpoint in", default="checkpoints")
parser.add_argument("--arch", help="Architecture to use for the pretrained model", default="densenet121")
parser.add_argument("--learning_rate", help="Learning Rate", type=float, default=0.003)
parser.add_argument("--hidden_units", help="Number of hidden units for the final output classifier", 
                    type=int, default=512)
parser.add_argument("--epochs", help="Number of epochs to train", 
                    type=int, default=3)
parser.add_argument("--gpu", help="Whether to use GPU for training and interference", action="store_true")

# Get the arguments
args = parser.parse_args()




def get_dataloaders(data_dir):
    """ Return dataloaders for train and validation datasets. Assume training and validation 
    files are in data_dir + '/train' and data_dir + '/valid'. Also return the class_to_idx. 
    """
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # Definetransforms for the training and validation
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    return trainloader, validloader, train_data.class_to_idx


def setup_training(architecture="densenet121", learning_rate=0.003, hidden_units=512):
    """ Return the model, optimizer, criteriorn based on the input parameters
    """
    
    model_cls = getattr(models, architecture)
    model = model_cls(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False


    #figure out the input size to the classifier with a fake input 
    model.classifier = nn.Sequential(nn.ReLU())
    test_input = torch.randn(1, 3, 224, 224)
    test_output = model.forward(test_input).view(1, -1)
    
    # Replace with classifier
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(test_output.shape[1], hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    model.classifier = classifier
    
    print(model)
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    
    return model, optimizer, criterion
    
def train(data_dir, save_dir, architecture, learning_rate, hidden_units, epochs, gpu):
    """ Train model using data from data_dir, with the passed architecture, learning_rate, and hidden units. 
    The training can happen on cpu or cuda based on gpu flag. Saves the trained model to save_dir. 
    """
    
    device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
    print(f"device:{device}")
    
    # Get training and validation loaders
    trainloader, validloader, class_to_idx = get_dataloaders(data_dir)
    
    # Get model, optimizer, and criterion
    model, optimizer, criterion = setup_training(architecture, learning_rate, hidden_units)
    
    # Train the model 
    steps = 0
    running_loss = 0
    print_every = 10 
    
    model = model.to(device)

    start = time.time()
    
    for epoch in range(epochs):

        # Train loop
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


            # Test on validation set
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader) * 100:.2f}%")
                running_loss = 0
                model.train()
                
    print(f"Finished training on {device}. Total training time: {time.time()-start:.2f} sec")
    
    # Save to checkpoint 
    save_path = os.path.join(save_dir, f'checkpoint_{architecture}_{datetime.datetime.now():%Y-%m-%d}.pth')
    checkpoint = { 'state_dict': model.state_dict(),
                  'arch': architecture,
                  'hidden_units': hidden_units,
                  'class_to_idx': class_to_idx }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")
    
    
if __name__ == "__main__":
    
    train(args.data_directory, args.save_dir, args.arch, 
          args.learning_rate, args.hidden_units, args.epochs, args.gpu)
