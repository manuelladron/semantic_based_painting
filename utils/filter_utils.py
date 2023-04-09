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
from losses import loss_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def filter_pixelwise(stroke, canvas, brush_size, num_params, renderer, device):
    """
    
    Renderer network takes in all stroke parameters except for RGB. Renderer outputs an alpha stroke of 128x128 with a white background. 
    Stroke black, background white 
    :param stroke: strokes that are being optimized, a tensor of shape [total patches, 13]
    :param canvas: all canvases patches, a tensor of shape [total patches, 3, 128, 128]
    
    """
    width = canvas.shape[2]
    i_til_rgb = 10
    
    # Make it opaque and clip width -> Transparency is being controlled by 
    #stroke = remove_transparency(stroke, device, num_params)
    stroke = utils.clip_width(stroke, num_params, max=brush_size, device=device)

    # Get stroke alpha 
    alpha = (1 - renderer(stroke[:, :i_til_rgb])).unsqueeze(1) # white stroke, black_background [n_patches, 128, 128] -> [n_patches, 1, 128, 128]
    
    # Multiply alpha by RGB params
    color_stroke = alpha * stroke[: , -3:].view(-1, 3, 1, 1) # [N, 3, 128, 128]

    # Reshape alpha 
    alpha = alpha.view(-1, 1, width, width) # [N, 1, 128, 128]
    
    # Blend 
    # 1 - alpha = 0 in the stroke regions, and 1 in the background. It zeroes out the canvas at the stroke region and adds canvas with color stroke, whose bckg = 0
    canvas = canvas * (1-alpha) + color_stroke 
    
    return canvas.clip(min=0, max=1)


def draw_red_square(I, x, y, R, color='red'):
    # Ensure that the input image is a tensor
    assert isinstance(I, torch.Tensor), "Input image should be a PyTorch tensor."
    
    # Ensure that the input coordinates and square size are integers
    assert isinstance(x, int) and isinstance(y, int) and isinstance(R, int), "Coordinates and square size should be integers."
    
    # Ensure that the input coordinates are within the image bounds
    if x >= 0 and x < I.shape[2] and y >= 0 and y < I.shape[1]: # "Coordinates should be within the image bounds."
        print("Coordinates should be within the image bounds.")
        print(f'x: {x}, y: {y}')
    
    # # Create a copy of the input image to draw on
    # I_draw = I.clone()
    
    # Compute the pixel coordinates of the square corners
    x1 = max(0, x - (R // 2))
    y1 = max(0, y - (R // 2))
    
    x2 = min(I.shape[3] - 1, x + (R // 2))
    y2 = min(I.shape[2] - 1, y + (R // 2))
    
    # Draw the red square on the image
    if color == 'red':
        I[:, :, y1:y2+1, x1:x2+1] = torch.tensor([1, 0, 0]).view(1, 3, 1, 1)

    elif color == 'blue':
        I[:, :, y1:y2+1, x1:x2+1] = torch.tensor([0, 0, 1]).view(1, 3, 1, 1)

    else:
        I[:, :, y1:y2+1, x1:x2+1] = torch.tensor([0, 1, 0]).view(1, 3, 1, 1)

    return I

def check_mask_overlap(A, B, threshold=0.5):
    """
    Check if non-zero values in binary mask B overlap with zero values in binary mask A.

    Args:
        A (torch.Tensor): binary mask, shape (H, W) -> binary segmentation mask 
        B (torch.Tensor): binary mask, shape (H, W) -> alpha brushstroke 
        threshold (float): overlap threshold, default 0.5

    Returns valid strokes:
        bool: True if non-zero values in B overlap with zero values in A by more than threshold, False otherwise
    """
    # Threshold the binary masks
    A = (A > 0).float()
    B = (B > 0).float()

    # Compute the overlap between A and B: since they are multiplying, the resulting overlap is between the white area of the binary mask and the white area of the stroke
    overlap = A * B # if stroke 1s falls entirely in a mask region 1s, this equals to 1. If falls entirely outside the mask, this equals to 0. 
    
    # If there is no overlap, return False
    if torch.sum(overlap) == 0:
        return False

    # Compute the fraction of B that overlaps with A with respect to the brush size 
    overlap_fraction = torch.sum(overlap) / torch.sum(B)

    # If the overlap fraction is less than the threshold, return True
    if overlap_fraction > threshold:
        return True
    
    else:
        return False


def filter_strokes(target_patches, strokes, mask, indices, brush_size, renderer, logger, device, mode, debug=False):
    """
    Filters out strokes that aren't in the mask 
    
    :target patches: a tensor of shape [N, 3, 128, 128] where N is >> M and is all the patches
    
    :strokes: optimized strokes of shape [budget, M, 13] <- M is the number of patches that have a mask 
    
    :mask: tensor of shape [M, 1, 128, 128] <- M is the number of patches corresponding to a mask 
    
    :indices: list of integers of len M, with indices corresponding to patches that have True values representing a mask 
    
    :returns : padded strokes of shape [budget, npatches, 13], a list with indices that indicate valid patches within the 35, and boolean tensor of shape [budget, npatches]
    """

    # Double check the number of patches that have been optimized correspond to the length of the indices list 
    assert strokes.shape[1] == len(indices) == mask.shape[0]
    strokes = strokes.permute(1, 0, 2) # [budget, M, 13] --> [M, budget, 13]

    num_params = strokes.shape[2]

    debug_strokes = [] # we are returning this 
    if debug:
        strokes_debug = strokes.clone()
    
    npatches = M = mask.shape[0] # patches corresponding to this segmentation mask 
    budget = strokes.shape[1]
    
    # Boolean tensor in the shape (M, budget) to indicate whether strokes are valid or not 
    valid_strokes_indices = torch.zeros(npatches, budget, dtype=torch.bool).to(device)
    valid_patch_indices = [] # list to cull out unvalid patches
    valid_strokes = []
    
    # Dumb tensor use for padding 
    dumb_tensor = (torch.ones(num_params) * -1).to(device)  # [13]
    
    total_non_valid = 0
    total_strokes = 0
    # Iterate over patches that belong to a mask 
    for i in range(npatches):

        # Set up patches 
        if mode == 'uniform':
            target_patch = target_patches[indices[i]].unsqueeze(0) # [3, 128, 128] select target_patch that correspond to segmentation mask 
        else:
            target_patch = target_patches
        
        strokes_patch = strokes[i] # [budget, 13]
        mask_patch = mask[i].squeeze() # [H, W]
        
        valid_strokes_patch = []
        nonvalid = 0
        
        if debug:
            print(f'patch idx: {indices[i]}')
            strokes_patch_debug = strokes_debug[i]
            mask_debug = mask_patch.unsqueeze(0).repeat(3, 1, 1)
            debug_strokes_patch = []
            
            # Visual debugging 
            new_tensor = torch.zeros(1, 3, 128, 128*3)
            new_tensor[:, :, :, 128:128*2] = mask_debug
            new_tensor[:, :, :, 128*2:] = target_patch

        # Iterate over all stroke parameters 
        for j in range(budget):
            total_strokes += 1
            canvas_debug, alpha = utils.forward_renderer(strokes_patch[j].unsqueeze(0), mask_patch.unsqueeze(0).unsqueeze(1), brush_size, num_params, renderer, device, return_alpha=True) # [1, 3, 128, 128]
            
            stroke = (strokes_patch[j].clip(0, 1) * 127).long() # indiviudal stroke 

            if num_params == 13:
                x0, y0 ,x, y, x2, y2, r0, r2, t0, t2, r, g, b = stroke # x == row, y == col (x is the index in the vertical axis, y is the index in the horizontal axis)
            else:
                x0, y0, x2, y2, r0, r2, t0, t2, r, g, b = stroke
            
            # Log for debug
            if debug:
                canvas_debug = draw_red_square(canvas_debug, y0.item(), x0.item(), R=5, color='red')
                canvas_debug = draw_red_square(canvas_debug, y.item(),  x.item(),  R=5, color='green')
                canvas_debug = draw_red_square(canvas_debug, y2.item(), x2.item(), R=5, color='blue')
                new_tensor[:, :, :, :128] = canvas_debug
                
                # For debugging - 
                stroke_debug = strokes_patch_debug[j]
                stroke_debug[-3:] = torch.cuda.FloatTensor([0., 1., 0.])
                
                print(f'\nStroke {j} --- x0: {x0}, y0: {y0}, x:{x}, y: {y}, x2: {x2}, y2: {y2}')
                print(f'Coords x0, y0 in mask: ', mask_patch[x0, y0].item())
                print(f'Coords x, y in mask: ', mask_patch[x, y].item())
                print(f'Coords x2, y2 in mask: ', mask_patch[x2, y2].item())
                
            # 1) Check for pixelwise overlap with mask  -> the higher the threshold the more strict is the algorithm to select valid strokes 
            valid_stroke = check_mask_overlap(mask_patch, alpha.squeeze(), threshold=0.98) # if where mask==1 and stroke==1 overlap for more than 80% then it's valid 
            
            # 2) Check for color distance: even if it's a valid stroke, if the color is off, discard it 
            # stroke = strokes_patch[j].clip(0, 1) 
            # rgb = stroke[-3:] 

            # color_true = target_patch.squeeze() * alpha  # [1, 3, 128, 128]
            # color_stroke = (alpha.unsqueeze(0) * rgb.view(3, 1, 1)).squeeze(0) # [1, 3, 128, 128]
            
            # distance_euclidean = torch.sqrt(torch.sum((color_stroke - color_true) ** 2))
            # distance_queen_wise = loss_utils.queen_wise_distance(color_stroke, color_true)
            # print(f'\ndistance euclidean: {distance_euclidean.item()}, queen_wise: {distance_queen_wise.item()}')


            # if distance_euclidean >= 1.5:
            #     valid_stroke = False
            #     print('COLOR NOT VALID\n')

            # if debug and valid_stroke:
            #     color_tensor = torch.zeros(1, 3, 128, 128*2)
            #     color_tensor[:, :, :, :128] = color_stroke
            #     color_tensor[:, :, :, 128:] = color_true

            #     logger.add_image(f'colors_id_{indices[i]}', img_tensor=color_tensor.squeeze(0), global_step=j)
            #     logger.add_scalar(f'colors_id_{indices[i]}_distance', distance_euclidean.item(), global_step=j)
                

            # Check if the start and end coordinates are in mask (not counting on x,y point because we are not applying the formula to make it middle pt)
            # Formula 
            # x1 = x0 + (x2 - x0) * x1
            # y1 = y0 + (y2 - y0) * y1

            if valid_stroke == False or mask_patch[x0, y0].item() == 0 or mask_patch[x2, y2].item() == 0:
                nonvalid += 1
                total_non_valid += 1
                valid_strokes_patch.append(dumb_tensor) # a form of padding 
                
                if debug:
                    stroke_debug[-3:] = torch.cuda.FloatTensor([1., 0., 0.]) # paint it red 
                    debug_strokes_patch.append(stroke_debug)
                    print(f'filter_id_{indices[i]}_stroke_{j}: NOT VALID')

                continue
            
            else:
                valid_strokes_indices[i, j] = True
                valid_strokes_patch.append(strokes_patch[j])
                
                if debug:
                    print(f'filter_id_{indices[i]}_stroke_{j}: VALID')
                    debug_strokes_patch.append(stroke_debug)
                    logger.add_image(f'valid_filter_id_{indices[i]}', img_tensor=new_tensor.squeeze(0), global_step=j)

        valid_strokes_patch = torch.vstack(valid_strokes_patch).to(device) # [all_strokes, 13]
        valid_strokes.append(valid_strokes_patch)

        if debug:
            print(f'non valid: {nonvalid}/{budget}')      
            debug_strokes_patch = torch.vstack(debug_strokes_patch).to(device) # [all_strokes, 13]
            debug_strokes.append(debug_strokes_patch)
        
        # if nonvalid is not the same as budget it means that at least there is one valid stroke 
        if nonvalid != budget:
            valid_patch_indices.append(indices[i]) # append only the patch that has strokes 
    
    if valid_strokes != []:
        valid_strokes = torch.stack(valid_strokes, dim=0).permute(1,0,2).to(device) # [npatches, budget, 13] -> [budget, npatches, 13]
        if debug:
            debug_strokes = torch.stack(debug_strokes, dim=0).permute(1,0,2).to(device) # [npatches, budget, 13] -> [budget, npatches, 13]
    
    else:
        print('there are no valid strokes in the entire optimization')

    valid_strokes_indices = valid_strokes_indices.permute(1,0) # [budget, npatches]

    print(f'\n TOTAL NON-VALID STROKES: {total_non_valid} / {total_strokes}')

    return valid_strokes, valid_patch_indices, valid_strokes_indices, debug_strokes


