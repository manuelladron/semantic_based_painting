import os 
import pdb 
import time
import numpy 
import torch 
from models.vgg_strotss import Vgg16_Extractor
from utils import utils 
from utils import render_utils as RU
from losses import clip_loss as CL
from losses import loss_utils
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def optimization_loop(args, src_img, opt_steps, target_patches, prev_canvas, strokes, budget, brush_size, patches_limits, npatches_w, level, mode, general_canvas, optimizer, renderer, num_params, logger, perc_net, opt_style=False):
    """
    :param opt_steps: optimization steps, an integer
    :param prev_canvas: canvas that correspond to the previous layer, a tensor of shape [N,C,H,W]
    :param strokes: stroke parameters to be optimized, a tensor of shape [n_patches, budget, 13]
    :param_brush_size: max brush size, used to clip the brush size, a float 
    :param patches_limits: a list of N patches limits in the shape ((h_st, h_end),(w_st, w_end))
    :param mode: painting mode: uniform or natural
    :param general_canvas: the general bigger canvas to compose the crops onto, a tensor of shape [1, 3, H, W]
    :param optimizer: pytorch optimizer 
    """
    
    # Optimization loop             
    for i in range(opt_steps):
        
        # Reset canvas at each iteration: Get previous canvas as canvas 
        canvas = prev_canvas # [N, C, H, W]
        
        # Get painting (this is by patches)
        canvas, canvases_seq, strokes_seq = utils.render(canvas, strokes, budget, brush_size, renderer, num_params)

        # If using global loss, get a smaller size general canvas and source image to compute loss later
        if args.global_loss or opt_style:
            general_canvas_dec, source_img_dec = utils.blend_all_canvases(canvas, patches_limits, general_canvas, src_img, logger, resize_factor=0.5)

        if i == 0:
            # Visualize a random patch 
            logger.add_image(f'patch_4', canvas[4].squeeze(), global_step=0)

        # Put patches together to compose the big general canvas 
        if i % 50 == 0 or i == (opt_steps - 1):
            st_bl = time.time()
            general_canvas = utils.compose_general_canvas(args, canvas, mode, patches_limits, npatches_w, general_canvas, blendin=True) # updates general canvas 
            end_bl = time.time()
            
            print(f'Blending time: {end_bl-st_bl} seconds')
            if opt_style:
                style_str = 'style'
            else:
                style_str = 'base'
            
            logger.add_image(f'general_painting_{style_str}_{level}', img_tensor=general_canvas.squeeze(), global_step=i)
            
            # if using global loss, log info
            if args.global_loss:
                logger.add_image(f'general_painting_dec_{level}', img_tensor=general_canvas_dec.squeeze(), global_step=i)
                if level == 0:
                    logger.add_image(f'src_img_dec_{level}', img_tensor=source_img_dec.squeeze(), global_step=i)
            
        # Compute loss and optimize 
        if opt_style == False:
            loss, l1_loss, perc_loss = loss_utils.compute_loss(args, perc_net, canvas, target_patches, use_clip=False)
            # length_loss = loss_utils.control_point_distance_loss(strokes, beta=0.1)
            # length_loss = loss_utils.differentiable_bezier_loss_batch(strokes, radius=0.1)
            #loss += length_loss 
        
        # Increasing lambdas for content clip loss 
        else:
            if level == 0 or level == 1:
                args.content_lambda = 10
            elif level == 2:
                args.content_lambda = 50
            elif level == 3:
                args.content_lambda = 80
            elif level == 4: 
                args.content_lambda = 150

            loss = loss_utils.compute_loss(args, perc_net, general_canvas_dec, source_img_dec, use_clip=True)
        
        # Compute global loss 
        if args.global_loss:
            loss, l1_loss, perc_loss = loss_utils.compute_loss(args, perc_net, general_canvas_dec, source_img_dec, use_clip=False)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add to logger 
        if i % 10 == 0:
            if opt_style:
                loss_utils.log_losses(logger, 'clip loss', loss, i, opt_steps, level)
            else:
                loss_utils.log_losses(logger, 'total loss', loss, i, opt_steps, level)
                loss_utils.log_losses(logger, 'l1 loss', l1_loss, i, opt_steps, level)
                #loss_utils.log_losses(logger, 'length loss', length_loss, i, opt_steps, level)
                if args.w_perc > 0:
                    loss_utils.log_losses(logger, 'perc loss', perc_loss, i, opt_steps, level)

    return canvas, general_canvas



def optimization_loop_boundaries(args, src_img, opt_steps, target_patches, prev_canvas, 
                                strokes, budget, brush_size, patches_limits, npatches_w, 
                                level, mode, general_canvas, optimizer, renderer, num_params, 
                                logger, perc_net, indices, global_loss=False, name='boundaries', mask=None):
    
    """
    :param opt_steps: optimization steps, an integer
    :param prev_canvas: canvas that correspond to the previous layer, a tensor of shape [N,C,H,W]
    :param strokes: stroke parameters to be optimized, a tensor of shape [n_patches, budget, 13]
    :param_brush_size: max brush size, used to clip the brush size, a float 
    :param patches_limits: a list of N patches limits in the shape ((h_st, h_end),(w_st, w_end))
    :param mode: painting mode: uniform or natural
    :param general_canvas: the general bigger canvas to compose the crops onto, a tensor of shape [1, 3, H, W]
    :param optimizer: pytorch optimizer 
    :param mask: list containing all binary mask patches 
    """
    
    # Out of all patches, select those that have boundaries/mask
    target_patches = torch.index_select(target_patches, 0, torch.Tensor(indices).int().to(device))
    
    # Mask is a list containing all patches, get only the binary masks given by the indices 
    if mask != None:
        bin_mask_patches = torch.cat(mask, dim=0).to(device)
        mask = torch.index_select(bin_mask_patches, 0, torch.Tensor(indices).int().to(device))
    
    # Generate indices that correspond to the canvases that do not have boundaries 
    total_number_of_indices = prev_canvas.shape[0]
    indices_no_boundaries = list(set(range(total_number_of_indices)) - set(indices))

    # Optimization loop             
    for i in range(opt_steps):
        
        # Reset canvas variable at each iteration: Get previous canvas as canvas 
        canvas = prev_canvas # [N, C, H, W] <- N = num_total_patches
        canvas_selected = torch.index_select(canvas, 0, torch.Tensor(indices).int().to(device)) # [M, C, H, W] M << N
        
        # Render strokes, which contain only patches that have masks or boundaries, so we need to select first only the canvases that have boundaries/masks 
        canvas_selected, _, _ = utils.render(canvas_selected, strokes, budget, brush_size, renderer, num_params)
        
        # Merge all canvases (mask and no mask) - update canvas variable 
        canvas = utils.merge_tensors(canvas_selected, canvas, indices, indices_no_boundaries) # [N, C, H, W]

        # If using global loss, get a smaller size general canvas and source image to compute loss later
        if global_loss:
            general_canvas_dec, source_img_dec = utils.blend_all_canvases(canvas, patches_limits, general_canvas, src_img, logger, resize_factor=0.5)

        # Put patches together to compose the big general canvas 
        if i % 50 == 0 or i == (opt_steps - 1):

            general_canvas = utils.compose_general_canvas(args, canvas, mode, patches_limits, npatches_w, general_canvas, blendin=True) # updates general canvas 
            logger.add_image(f'general_painting_{name}_{level}', img_tensor=general_canvas.squeeze(), global_step=i)
            
            # if using global loss, log info
            if global_loss:
                logger.add_image(f'{name}_dec_{level}', img_tensor=general_canvas_dec.squeeze(), global_step=i)

        # if i % 100 == 0:
        #     # Visualize progress 
        #     utils.visualize_progress(canvas, canvas_selected, general_canvas, mask, indices)

        # Compute loss between canvases with boundaries and boundaries patches and optimize 
        loss, l1_loss, perc_loss = loss_utils.compute_loss(args, perc_net, canvas_selected, target_patches.float(), use_mse=True, use_clip=False, mask=mask)
        
        # Use length loss only on masks not on boundaries 
        #if name != 'boundaries' and mask != None:
            #length_loss = loss_utils.control_point_distance_loss(strokes, beta=0.1)
            #length_loss = loss_utils.differentiable_bezier_loss_batch(strokes, radius=0.1)
            #loss += length_loss 
        
        if global_loss:
            loss, l1_loss, perc_loss = loss_utils.compute_loss(args, perc_net, general_canvas_dec, source_img_dec, use_clip=False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add to logger 
        if i % 10 == 0:
            loss_utils.log_losses(logger, 'total loss', loss, i, opt_steps, level)
            loss_utils.log_losses(logger, 'l1 loss', l1_loss, i, opt_steps, level)
            # if name != 'boundaries' and mask != None:
            #     loss_utils.log_losses(logger, 'length loss', length_loss, i, opt_steps, level)
            if args.w_perc > 0:
                loss_utils.log_losses(logger, 'perc loss', perc_loss, i, opt_steps, level)


    return canvas, general_canvas




