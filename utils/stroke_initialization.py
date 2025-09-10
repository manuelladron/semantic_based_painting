import os 
import pdb 
import json 
import numpy as np 
import torch 
import cv2 
from PIL import Image
from skimage.segmentation import find_boundaries
from models.renderers import FCN
from torch.utils.tensorboard import SummaryWriter
import torch 
import torch.nn.functional as F
import torchvision
from colour import Color
from src.segmentation import segment_image
from utils import render_utils as RU
from utils import utils 

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 
device = "mps" if torch.backends.mps.is_available() else "cpu"
# Set default tensor type to float32
torch.set_default_dtype(torch.float32)

####################### For straight brush 
def generate_random_batch(batch_size):
    x = np.random.rand(batch_size)
    power = 2
    y = np.power(x, power)
    mask = np.random.rand(batch_size) <= y
    x[mask] = 1.0
    return x

def generate_random_batch_torch(batch_size):
    x = torch.rand((batch_size,))
    power = 2
    y = torch.pow(x, power)
    mask = torch.rand((batch_size,)) <= y
    x[mask] = 1.0
    return x.to(device)


def generate_random_params_straight(batch_size):
    
    start = np.random.rand(batch_size, 2)
    end = np.random.rand(batch_size, 2)

    start_thickness = np.random.rand(batch_size, 1)
    #end_thickness = np.random.rand(batch_size, 1)
    
    mu = 0
    sigma = 0.2

    # Generate random samples from normal distribution
    end_thickness = start_thickness + np.random.normal(mu, sigma, size=(batch_size, 1))

    # Constrain range of end_thickness to [0, 1]
    end_thickness = np.clip(end_thickness, 0, 1)

    start_alpha = generate_random_batch(batch_size).reshape(-1, 1)
    end_alpha = generate_random_batch(batch_size).reshape(-1, 1)

    params = np.concatenate([start, end, start_thickness, end_thickness, start_alpha, end_alpha], axis=1)
    params = torch.tensor(params, dtype=torch.float32)
    
    return params


def create_grid_strokes(budget, num_params, patch, device, std = 0.2):
    """
    Creates a grid of strokes evenly distributed on the canvas. 
    
    Each stroke is parameterized by a 13D vector [x0,y0,x1,y1,x2,y2, r0, r2, t0, t2, rgb] 
    
    Idea: 1) get midpoints of each stroke, spread them evenly on the canvas 
          2) Have a gaussian centered on each of the midpoints, and sample start and end points 
          3) Other parameters (color, thickness, etc, are randomly sampled)
    """
    # Get a square number 
    strokes_per_side = int(np.sqrt(budget)) # sqrt(300) ~ 17
    #print('strokes_per side: ', strokes_per_side)

    # Arrange midpoint coordinates on canvas 
    x1 = torch.arange(0.05, 0.95, 1 / strokes_per_side).to(device)
    y1 = torch.arange(0.05, 0.95, 1 / strokes_per_side).to(device)

    # Create meshgrid 
    x, y = torch.meshgrid(x1, y1) # x -> [strokes_per_side, strokes_per_side], y -> [strokes_per_side, strokes_per_side]

    # Flatten these coordinates 
    x = torch.flatten(x).unsqueeze(1) # [budget, 1]
    y = torch.flatten(y).unsqueeze(1) # [budget, 1]

    # Get x0,y0 and x2,y2 coordinates 
    x0 = torch.normal(mean=x, std=std).clip(min=0, max=1).to(device)
    y0 = torch.normal(mean=y, std=std).clip(min=0, max=1).to(device)

    x2 = torch.normal(mean=x, std=std).clip(min=0, max=1).to(device)
    y2 = torch.normal(mean=y, std=std).clip(min=0, max=1).to(device)

    # for curved strokes 
    if num_params == 13:
        # Other parameters: radius, transparency and rgb 
        rad = torch.rand(budget, 2, requires_grad=True, device=device)    # [budget, 2]
        transp = torch.ones(budget, 2, requires_grad=True, device=device) # [budget, 2]
        #rgb = torch.rand(budget, 3, requires_grad=True, device=device)

        # Convert patch to a color palette 
        #palette = [(205,180,219),(255, 200, 221),(255, 175, 204),(189, 224, 254),(162, 210, 255)]
        #patch = color_palette_replace(patch, palette)
        
        # print('x', x.shape) # [budget, 1]
        # print('y', y.shape) # [budget, 1]
        # print('x_trans', (x*128).squeeze().long().shape)
        # print('patch: ', patch[:, (x*128).squeeze().long(), (y*128).squeeze().long()].shape)
        
        #pdb.set_trace()
        rgb = (torch.ones(budget, 3, requires_grad=True, device=device) * (patch[:, (x*128).squeeze().long(), (y*128).squeeze().long()]).permute(1,0)).float() # [budget, 3]
        #print('rgb shape: ', rgb.shape)
        #pdb.set_trace()
        # Put them together 
        strokes = torch.cat([x0, y0, x, y, x2,y2, rad, transp, rgb], dim=1).requires_grad_()

    # For straight strokes 
    else:
        start_thickness = torch.rand(budget, 1).to(device)
        
        mu = 0
        sigma = 0.2
        # Generate random samples from normal distribution
        end_thickness = start_thickness + torch.normal(mean=torch.tensor(mu), std=torch.tensor(sigma), size=(budget, 1)).to(device)

        # Constrain range of end_thickness to [0, 1]
        end_thickness = torch.clip(end_thickness, 0, 1)

        # Transparency 
        start_alpha = generate_random_batch_torch(budget).reshape(-1, 1)
        end_alpha = generate_random_batch_torch(budget).reshape(-1, 1)

        rgb = (torch.ones(budget, 3, requires_grad=True, device=device) * (patch[:, (x*128).squeeze().long(), (y*128).squeeze().long()]).permute(1,0)).float()

        # Put them together 
        strokes = torch.cat([x0, y0, x2,y2, start_thickness, end_thickness, start_alpha, end_alpha, rgb], dim=1).requires_grad_()

    return strokes 

def init_strokes(budget, mode, device, target_patch, num_params=13, std=0.2):
    """
    Initializes random strokes parameters on a canvas given a budget
    """
    if mode == 'random':
        # uniform distribution 
        strokes = torch.rand(budget, num_params, requires_grad=True, device=device)
    
    elif mode == 'grid':
        strokes = create_grid_strokes(budget, num_params, target_patch, device, std=std)

    return strokes 

def init_boundary_stroke_params(boundaries, patch, budget, num_params, device, std = 0.05):
    """
    Given a boolean tensor of shape [128x128], get x,y coordinates over True values, and build the rest of the stroke based on this
    :param boundaries: a boolean tensor of shape [128x128]
    :budget: number of strokes 

    :returns: strokes parameter 
    """
    x, y = init_coordinates_in_mask(boundaries, budget, device) # Tensor Long [budget]

    # Convert to float 
    x = (x/128.0).unsqueeze(1) # [budget, 1]
    y = (y/128.0).unsqueeze(1)

    # Given x and y, both long tensors of shape [budget] representing locations within a patch (range 0-127), get x0 and x2 sampling from a gaussian.
    x0 = torch.normal(mean=x, std=std).clip(min=0, max=1).to(device)
    y0 = torch.normal(mean=y, std=std).clip(min=0, max=1).to(device)

    x2 = torch.normal(mean=x, std=std).clip(min=0, max=1).to(device)
    y2 = torch.normal(mean=y, std=std).clip(min=0, max=1).to(device)

    if num_params == 13:
        # Other parameters: radius, transparency and rgb 
        rad = torch.rand(budget, 2, requires_grad=True, device=device)
        transp = torch.ones(budget, 2, requires_grad=True, device=device)
        
        # Convert patch to a color palette 
        #palette = [(205,180,219),(255, 200, 221),(255, 175, 204),(189, 224, 254),(162, 210, 255)]
        #patch = color_palette_replace(patch, palette)

        rgb = (torch.ones(budget, 3, requires_grad=True, device=device) * (patch[:, (x*128).squeeze().long(), (y*128).squeeze().long()]).permute(1,0)).float()
        #rgb = torch.rand(budget, 3, requires_grad=True, device=device)
        # Put them together 
        strokes = torch.cat([x0,y0, x, y, x2,y2, rad, transp, rgb], dim=1).requires_grad_()

    else:
        start_thickness = torch.rand(budget, 1).to(device)
        mu = 0
        sigma = 0.2
        # Generate random samples from normal distribution
        end_thickness = start_thickness + torch.normal(mean=torch.tensor(mu), std=torch.tensor(sigma), size=(budget, 1)).to(device)

        # Constrain range of end_thickness to [0, 1]
        end_thickness = torch.clip(end_thickness, 0, 1)

        # Transparency 
        start_alpha = generate_random_batch_torch(budget).reshape(-1, 1)
        end_alpha = generate_random_batch_torch(budget).reshape(-1, 1)

        rgb = (torch.ones(budget, 3, requires_grad=True, device=device) * (patch[:, (x*128).squeeze().long(), (y*128).squeeze().long()]).permute(1,0)).float()

        # Put them together 
        strokes = torch.cat([x0, y0, x2,y2, start_thickness, end_thickness, start_alpha, end_alpha, rgb], dim=1).requires_grad_()
    
    return strokes 

def init_coordinates_in_mask(A, N, device):
    """
    Given a binary tensor A of shape [H, W] and an integer N, returns x,y random coordinates over True values of A. 
    :param A: mask tensor of shape [H, W]
    :param N: integer
    """
    H, W = A.shape # A has a shape [H, W]
    indices = torch.nonzero(A).t() # Indices of the True values in A, shape is [2, num_True_values]
    
    if indices.shape[1] == 0:
        raise ValueError("Tensor A does not contain any True values.")
    
    # Pass torch.ones with length indices to torch.multinomial to have equal probability for each index. 
    selected_indices = torch.multinomial(torch.ones(indices.shape[1]), N, replacement=True) # Sample N indices from the True values in A, shape is [N]
    selected_coordinates = indices[:, selected_indices].t() # Select N random (x,y) pairs, shape is [N, 2]
    x = selected_coordinates[:, 0].to(device) # First column represents x-coordinates, shape is [N]
    y = selected_coordinates[:, 1].to(device) # Second column represents y-coordinates, shape is [N]
    
    return x, y

def init_strokes_patches(budget, mode, device, npatches, target_patches, std=0.2, num_params=13):
    strokes_l = []
    for p in range(npatches):
        strokes = init_strokes(budget, mode, device, target_patches[p], std=std, num_params=num_params)
        strokes_l.append(strokes)
    
    return torch.stack(strokes_l, dim=1).detach().requires_grad_() # [budget, npatches, 13]


def init_strokes_with_mask(N, budget, device, mask_or_edges, patches_limits_list, patches, edges=False, name='', num_params=13, std=0.05):
    """
    Initializes b-budget strokes (2-4 max) only on patches that have masks (or boundaries) (true values)
    
    :param N: number of pixels that represent boundaries.
    
    :param budget: integer with number of strokes to initialize 
    
    :mask_or_edges: alpha/binary mask (or edges), a tensor of shape [total_num_patches, 1, H, W]
    
    :patches_limits_list: a list with total_num_patches tensors 

    return strokes parameters of shape [budget, n_patches, num_params], and indices 
    """
    # Within mask patches, filter out those that have less than N pixels set to True 
    indices, selected_tensors, n_total = utils.select_tensors_with_n_true(mask_or_edges, patches_limits_list, N)
    strokes_l = []
    
    for i in range(len(patches_limits_list)):
        if i in indices:
            strokes = init_boundary_stroke_params(mask_or_edges[i].squeeze(), patches[i], budget, num_params, device, std=std)
            strokes_l.append(strokes)
    
    strokes = torch.stack(strokes_l, dim=1).detach().requires_grad_() # [budget, npatches, 13]
    
    return strokes, indices 



def color_palette_replace(image, palette):
    # Convert the tensor to a numpy array
    
    np_img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # Calculate the distance between each pixel in the image and each color in the palette
    dist = np.sum((np_img[:, :, np.newaxis, :] - palette) ** 2, axis=3)

    # Find the index of the closest color in the palette for each pixel in the image
    idx = np.argmin(dist, axis=2)

    # Replace the original RGB values in the image with the closest palette color
    new_img = np.array(palette)[idx]

    # Convert the numpy array back to a tensor
    new_img = torch.from_numpy(new_img).permute(2, 0, 1).to(device)

    # Return the new image tensor
    return new_img/255.
