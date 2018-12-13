from PIL import Image
import numpy as np
import torch

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


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with Image.open(image_path) as img:
        
        model = model.to(device)
        img_tensor = torch.unsqueeze(process_image(img),0).to(device)
        
        # Pass the img_tensor through the model and get the top k classes 
        logps = model.forward(img_tensor)
        ps = torch.exp(logps)
        top_p, top_class_idx = ps.topk(topk, dim=1)

        # Convert top_class_idx to actual class        
        top_class = [idx_to_class[idx] for idx in top_class_idx.squeeze().tolist()]
            
        return top_p.squeeze().tolist(), top_class