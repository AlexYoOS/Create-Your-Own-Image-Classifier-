# Imports
import argparse
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms as T, models, datasets
from torch.utils.data import DataLoader
from collections import OrderedDict
from PIL import ImageFile

# Create argument parser
def input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('dir', type=str, 
                        help='path to folder of images')
    parser.add_argument('--arch', type=str, default='resnet50', 
                        help='chosen model')
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--learn_rate', '-lr', type=float, default=0.001,
                        help='choose learning rate')
    parser.add_argument('--epochs', type=int, default=1,
                         help='numbers of epochs training')
    parser.add_argument('--save', type=str, default='checkpoint.pth',
                        help='save checkpoint file name')   
    parser.add_argument('--gpu', action="store_true",
                         help= 'gpu training option')

    # Returns argument parser
    return parser.parse_args()



# Data preperation wrapped function
def data_loading(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = T.Compose([T.Resize(256),
                                       T.RandomCrop(224),
                                       T.ColorJitter(brightness=.5, hue=.3),
                                       T.ToTensor(),
                                       T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_valid_transforms = T.Compose([T.Resize(256),
                                       T.CenterCrop(224),
                                       T.ToTensor(),
                                       T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets = {"train":datasets.ImageFolder(train_dir, train_transforms),
                  
                "valid":datasets.ImageFolder(valid_dir, test_valid_transforms),
                  "test":datasets.ImageFolder(test_dir, test_valid_transforms)}


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {"train":DataLoader(image_datasets["train"],batch_size=32, shuffle=True),
              "valid":DataLoader(image_datasets["valid"],batch_size=16, shuffle=True),
               "test":DataLoader(image_datasets["test"],batch_size=16, shuffle=True)}
    
    class_to_idx = image_datasets['train'].class_to_idx
    
    return dataloaders, class_to_idx

# Building Model for two different architectures
def build_model(arch, hidden_units):
    # Build conditional statments for two architectures
    if "squeeze" in arch:
        print("squeezenet is loading...")
        model = models.squeezenet1_1(weights='IMAGENET1K_V1')
        for param in model.parameters():
            param.requires_grad = False

        model.classifier._modules["1"] = nn.Conv2d(hidden_units, 102, kernel_size=(1, 1))
        model.num_classes = 102

        # Apply LogSoftmax to the output
        model.classifier._modules["2"] = nn.LogSoftmax(dim=1)
        
    else:
        print("resnet50 is loading...")
        model = models.resnet50(weights='IMAGENET1K_V2')
        for param in model.parameters():
            param.requires_grad = False
        
        num_ftrs  =  model.fc.in_features
        model.fc  = nn.Sequential(OrderedDict([
                          ('ll1', nn.Linear(num_ftrs, hidden_units)),
                          ('activation', nn.ReLU()),
                          ('ll2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    return model
    
# Training pass from Notebook
def training_model(model, lr, gpu, epochs, dataloaders):
    # Defining optimizer and loss function
    criterion=nn.NLLLoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)
    # training pass
    device = torch.device('mps' if torch.backends.mps.is_available() and gpu else 'cpu')

    # Image compatibility of truncated images in PIL
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # specifying epochs
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in dataloaders["train"]:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            model.to(device)

            optimizer.zero_grad()
            
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders["valid"]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        _, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"validation loss: {test_loss/len(dataloaders['valid']):.3f}.. "
                    f"validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
            model.train()
            
    return model, optimizer

# save function
def saving(arch, hidden_units, model, optimizer, epochs, lr, class_to_idx, save):
    
    # create a checkpoint
    checkpoint = {"arch": arch,
                  "hidden_units": hidden_units,
                  "state_dict": model.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "epochs": epochs,
                  "lr": lr,
                  "class_to_idx": class_to_idx}
    torch.save(checkpoint, save)

# Main program function defined below
def main():

    # Call argument parser function
    args = input_args()
    # Load data with arg parser data directory argument
    dataloaders, class_to_idx = data_loading(args.dir)
    # Load model
    model = build_model(args.arch, args.hidden_units)
    # Train model with customized model and loaded data
    model, optimizer = training_model(model, args.learn_rate, args.gpu, args.epochs, dataloaders)
    # saving model to checkpoint
    saving(args.arch, args.hidden_units, model, optimizer, args.epochs, args.learn_rate, class_to_idx, args.save)    
    

# Call to main function to run the program
if __name__ == "__main__":
    main()