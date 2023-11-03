import os 
import pdb 
import time
import numpy 
import torch 
import torch.nn.functional as F
from losses import clip_loss as CL


def log_losses(logger, loss_name, loss, global_step, total_steps, level):
    logger.add_scalar(f'{loss_name}_{level}', loss, global_step=global_step)
    print(f'{loss_name}: {loss.item()} | {global_step}/{total_steps} done!')


def compute_loss(args, perc_net, canvas, target_patches, use_mse = False, use_clip=False, mask=None):
    """
    We do all patches together
    :param canvas: [npatches, 3, 128, 128]
    :param target_patches: [npatches, 3, 128, 128]

    :return: loss
    """
    loss = 0.0
    
    if use_clip == False:
        
        if use_mse:
            if mask != None:
                #neg_weights = 2 * mask - 1 # shifts a mask of 0. 1. to a mask of -1. 1 and then flips it to 1. and -1
                l1_loss = torch.nn.MSELoss(reduction='none')(canvas, target_patches) 
                l1_loss = (l1_loss * mask).mean()
                
            else:
                l1_loss = torch.nn.MSELoss()(canvas, target_patches)
        else:
            # L1 loss 
            if mask != None:
                l1_loss = (F.l1_loss(canvas, target_patches, reduction='none') * mask).mean()
            else:
                l1_loss = F.l1_loss(canvas, target_patches, reduction='mean')
        loss += l1_loss
        
        # Perc loss 
        perc_loss = 0.0
        
        if args.w_perc > 0:
            feat_canvas = perc_net(canvas)  # [npatches, 3, 128, 128]
            feat_target = perc_net(target_patches)  # [npatches, 3, 128, 128]

            perc_loss = 0
            for j in range(len(feat_target)):
                perc_loss -= torch.cosine_similarity(feat_canvas[j], feat_target[j], dim=1).mean()
            perc_loss /= len(feat_target)

            loss += (perc_loss * args.w_perc)

        return loss, l1_loss, perc_loss

    else:
        clip_loss = CL.get_clip_loss(args, args.style_prompt, canvas, target_patches, use_patch_loss=True)
        return clip_loss


def bezier_loss(strokes, radius=0.1):

    x0, y0, x1, y1, x2, y2, r0, r2, t0, t2, r, g, b = torch.chunk(strokes, 13, dim=2)
    """
    Compute the bezier loss for the given start, middle, and end points.

    The bezier loss penalizes the distance between the start, middle, and end points
    of the bezier curve, in order to encourage deviation from a circular shape.

    Args:
        x0 (torch.Tensor): start x-coordinate, shape (B,)
        y0 (torch.Tensor): start y-coordinate, shape (B,)
        x1 (torch.Tensor): middle x-coordinate, shape (B,)
        y1 (torch.Tensor): middle y-coordinate, shape (B,)
        x2 (torch.Tensor): end x-coordinate, shape (B,)
        y2 (torch.Tensor): end y-coordinate, shape (B,)
        radius (float): penalty radius, default 0.1

    Returns:
        torch.Tensor: the bezier loss, shape (B,)
    """
    # Compute the distances between the start, middle, and end points
    d01 = torch.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    d12 = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    d02 = torch.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2)

    # Compute the loss for each example in the batch
    loss = torch.zeros((strokes.shape[0], strokes.shape[1]), dtype=torch.float32)
    loss[((d01 < radius) | (d12 < radius) | (d02 < radius)).any(dim=2)] = 1

    pdb.set_trace()
    return loss

def differentiable_bezier_loss_batch(strokes, radius=0.4):
    """Compute the differentiable bezier loss for a batch of stroke parameters.
    
    Args:
        strokes (torch.Tensor): A tensor of shape [num_strokes, num_patches, 13] containing
            the stroke parameters. The third dimension is a 13-D tuple which is defined by
            x0, y0, x1, y1, x2, y2, r0, r2, t0, t2, r, g, b.
        radius (float): The radius of the circular region around the start, middle and end points.
    
    Returns:
        torch.Tensor: A scalar tensor representing the differentiable bezier loss.
    """
    num_strokes, num_patches, _ = strokes.shape
    
    # Extract the start, middle, and end points
    p0 = strokes[:, :, :2]
    p1 = strokes[:, :, 2:4]
    p2 = strokes[:, :, 4:6]
    
    # Compute the distances between the start, middle, and end points
    d01 = torch.norm(p1 - p0, dim=-1)
    d12 = torch.norm(p2 - p1, dim=-1)
    d02 = torch.norm(p2 - p0, dim=-1)
    
    # Create a mask of 0's and 1's indicating which strokes and patches violate the radius constraint. If distance is less than the radius constraint, then adds a penalty
    mask = (d01 < radius) | (d12 < radius) | (d02 < radius)
    
    # Compute the differentiable loss as the mean of the mask
    loss = torch.mean(mask.type(torch.float32))
    
    #pdb.set_trace()
    return loss


def distance(p1, p2):
    """Compute the Euclidean distance between two points."""
    return torch.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def control_point_distance_loss(strokes, beta=1):
    """Loss function that penalizes small distances between control points."""
    # Extract control points from parameters tuple
    # Extract the start, middle, and end points
    p0 = strokes[:, :, :2]
    p1 = strokes[:, :, 2:4]
    p2 = strokes[:, :, 4:6]
    
    # Compute distances between control points
    d01 = distance(p0, p1)
    d12 = distance(p1, p2)
    
    # Penalize small distances with a linear function
    loss = (beta * (1/d01 + 1/d12)).mean()
    
    return loss


def minkowski5_distance(x, y):
    # Reshape tensors to have shape [N, C, L], where N=1, C=3, L=128*128
    x = x.view(1, -1, 128*128)
    y = y.view(1, -1, 128*128)

    # Compute the Minkowski distance with p=5
    distance = (torch.abs(x - y)**5).sum(dim=2)**(1/5)

    return distance.mean()

def squared_chord_distance(x, y):
    # Reshape tensors to have shape [N, L], where N=1, L=3*128*128
    x = x.view(1, -1)
    y = y.view(1, -1)

    # Normalize the vectors to have unit length
    x_norm = x / torch.norm(x, p=2)
    y_norm = y / torch.norm(y, p=2)

    # Compute the squared Euclidean distance between the normalized vectors
    distance = torch.sum((x_norm - y_norm)**2)

    return distance

def queen_wise_distance(x, y):
    # Reshape tensors to have shape [N, L], where N=1, L=3*128*128
    x = x.view(1, -1)
    y = y.view(1, -1)

    # Compute the Queen-wise distance
    dx = torch.abs(x - y)
    distance = torch.max(dx // 128, dx % 128).sum()

    return distance

def canberra_distance(x, y):
    # Reshape tensors to have shape [N, L], where N=1, L=3*128*128
    x = x.view(1, -1)
    y = y.view(1, -1)

    # Compute the Canberra distance
    numerator = torch.abs(x - y)
    denominator = torch.abs(x) + torch.abs(y)
    distance = torch.sum(numerator / denominator)

    return distance