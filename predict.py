# Imports 
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
from torchvision import transforms as T, models
import json
from collections import OrderedDict
from PIL import Image

# Create argument parser
def input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('single_image', type=str, help='path to the image that should be predicted')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth',
                        help='checkpoint file name')
    parser.add_argument('--topk', type=int, default=5,
                        help='specify number of output predictions')
    parser.add_argument('--flower_map', type=str, default='cat_to_name.json',
                        help='class to flower-species map')    
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='gpu training option')

    # Returns argument parser
    return parser.parse_args()

# load train model from checkpoint
def load_checkpoint(filepath):
    # load checkpoint
    checkpoint = torch.load(filepath)
    
    hidden_units = checkpoint['hidden_units']
    # conditional statement for chosen model
    if "squeeze" in checkpoint['arch']:
        print("squeezenet is loading from checkpoint...")

        model = models.squeezenet1_1(weights='IMAGENET1K_V1')
        for param in model.parameters():
            param.requires_grad = False

        model.classifier._modules["1"] = nn.Conv2d(hidden_units, 102, kernel_size=(1, 1))
        model.num_classes = 102

        # Apply LogSoftmax to the output
        model.classifier._modules["2"] = nn.LogSoftmax(dim=1)
    else:
        print("resnet50 is loading from checkpoint...")
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
     
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array     
    '''
    # Open Image
    pil_image = Image.open(image)
    
    # Determine if height or width are shortest side
    desired_size = (256, int(256 * (pil_image.size[1] / pil_image.size[0]))) \
    if pil_image.size[0] < pil_image.size[1] \
    else (int(256 * (pil_image.size[0] / pil_image.size[1])), 256)
    
    # Apply resizing on shortest side
    pil_image.thumbnail(desired_size)
    
   # Calculate the coordinates for cropping
    left = (pil_image.width - 224) / 2
    top = (pil_image.height - 224) / 2
    right = (pil_image.width + 224) / 2
    bottom = (pil_image.height + 224) / 2

    # Crop the image
    pil_image = pil_image.crop((left, top, right, bottom))
    
    # Define the normalization transform
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Convert image to tensor and transposes image to color-channel as first dimension
    image_tensor = T.ToTensor()(pil_image)

    # Return normalized image
    return normalize(image_tensor)

def predict(image_path, gpu, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file
    img = process_image(image_path)
    
    # Device setting
    device = torch.device('mps' if torch.backends.mps.is_available() and gpu else 'cpu')
    
    # Convert to 4D tensor and set to device
    img = img.unsqueeze(0)
    img = img.to(device)
    model.to(device)
    # Set model to eval mode
    model.eval()
    
    # Forward pass
    logps = model(img)
    # Exponantiate to get probabilities 
    preds = torch.exp(logps).data
    # Use get most likely predictions with top K
    probs, indices = preds.topk(topk)
    
    # Convert to numpy
    probs, indices = probs.data.cpu().numpy()[0], indices.data.cpu().numpy()[0]
    
    # Convert indices to classes with class_to_idx map
    idx_to_class = {x: y for y, x in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
    
    return probs, classes

def flower_mapping(flower_map, top_classes, probabilities):
    
    # open flower name file
    with open(flower_map, 'r') as file:
        flower_map = json.load(file, strict=False)
    
    # map classes to flower names
    flower_names = [flower_map[str(x)] for x in top_classes]
       
    # Print flower names and their probabilities   
    print(flower_names)
    print(probabilities)
    
    # After adding all the functions we would create a top level script with main()

def main():

    # Call argument parser function
    args = input_args()
    # load the model from checkpoint with argparser argument
    # class_to_idx item already attached to model
    model = load_checkpoint(args.checkpoint_path)
    # use model to predict single_image
    probabilities, top_classes = predict(args.single_image, args.gpu, model, args.topk)
    # Print predicted flower names and probabilities
    flower_mapping(args.flower_map, top_classes, probabilities)
    
    
# Top level script module:
if __name__ == "__main__":
    main()    