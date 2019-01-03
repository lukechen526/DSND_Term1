import argparse
import json
import os
from collections import OrderedDict

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import models

parser = argparse.ArgumentParser()

# required arguments

parser.add_argument('image_path', help="Path to the image to run inference on")
parser.add_argument('checkpoint_path', help="Path to checkpoint file", nargs="?", default="checkpoints/checkpoint.pth")
# optional arguments

parser.add_argument("--top_k", help="Show the top K classes",  type=int, default=5)
parser.add_argument("--gpu", help="Whether to use GPU for inference", action="store_true")
parser.add_argument("--category_names", help="Path to category name mapping file", type=str)

# Get the arguments
args = parser.parse_args()

    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Torch Tensor
    '''
    
    width, height = image.size
    
    # Re-scale so the shortest side is 256 px 
    if width >= height:
        resized_img = image.resize((int(width/height*256), 256))
    else:
        resized_img = image.resize((256, int(height/width*256)))
    
    w, h = resized_img.size
    
    # Center crop to 224 x 224 
    cropped_img = resized_img.crop((w//2 - 224//2, h//2 - 224//2, w//2 + 224//2, h//2 + 224//2))
    
    # Convert to numpy array
    np_image = np.asarray(cropped_img).astype(np.float64) / 255
    
    # Normalize: subtract mean and divide by std
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])    
    
    # Move the channel dimension to first dimension 
    np_image = np.transpose(np_image, (2, 0, 1))
    
    
    return torch.from_numpy(np_image).type(torch.FloatTensor)


def predict(image_path, ckpt_path, topk=5, gpu=True, cat_to_name=''):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
    
    # Load checkpoint filee
    checkpoint = torch.load(ckpt_path)
    architecture = checkpoint['arch']
    state_dict = checkpoint['state_dict']
    class_to_idx = checkpoint['class_to_idx']
    hidden_units = checkpoint['hidden_units']
    
    # Re-create the model and classifier based on checkpoint 
    
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
    model.load_state_dict(state_dict)
    
    
    
    # Invert class_to_idx 
    idx_to_class = {val: key for key, val in class_to_idx.items()}

    with Image.open(image_path) as img:
        
        model = model.to(device)
        model.eval()
        img_tensor = torch.unsqueeze(process_image(img),0).to(device)
        
        # Pass the img_tensor through the model and get the top k classes 
        logps = model.forward(img_tensor)
        ps = torch.exp(logps)
        top_p, top_class_idx = ps.topk(topk, dim=1)

        top_class_idx = top_class_idx.squeeze().tolist()
        
        if not isinstance(top_class_idx, list):
            top_class_idx = [top_class_idx]
        
        # Convert top_class_idx to actual class        
        top_class = [idx_to_class[idx] for idx in top_class_idx]
        
        # Convert class index to class name if cat_to_name is present
        
        if cat_to_name:
            top_class = [cat_to_name[cat] for cat in top_class]

        return top_p.squeeze().tolist(), top_class
    
    
    
if __name__ == "__main__":
    
    # Load category_names file if present 
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        cat_to_name =''
        
    top_p, top_class = predict(args.image_path, args.checkpoint_path, args.top_k, args.gpu, cat_to_name)
    print(top_p)
    print(top_class)
        
        