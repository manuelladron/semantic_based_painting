import numpy as np 
import torch 
import utils 
import pdb 
import torch.nn.functional as F
#import morphology
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def _forward_pass_l2p_alpha(action, num_params, renderer):
        action = action.view(-1, 10 + 3)  # [480, 13]
        action = utils.remove_transparency(action, num_params)
        stroke = 1 - renderer(action[:,:10])
        stroke = stroke.view(-1, 128, 128, 1)  # [480, 128, 128, 1]
        stroke = stroke.permute(0, 3, 1, 2)  # [480, 1, 128, 128]
        stroke = stroke.view(-1, 1, 128, 128)  # [batch, 1, 128, 128]

        return stroke.repeat(1,3,1,1) # returns alpha with 3 channels

def texturize(strokes, canvases, brush_size, t, num_params, renderer, args, level, start_organic_level, writer, onehot=None, mask_patches=None, painter=None):
    """
    Processes textures stroke by stroke
    :param strokes: [n_patches, 13]
    :param canvases: [n_patches, 3, 128, 128]
    onehot is [npatches]
    :return:
    """
    valid_strokes = 0
    invalid_strokes = 0
    strokes = utils.remove_transparency(strokes, num_params)
    strokes = utils.clip_width(strokes, num_params, max=brush_size)
    # iterate over patches and apply texture one by one
    alphas, foregrounds = [], []
    # pdb.set_trace()

    for p in range(strokes.shape[0]): # Iterate over patches. Single stroke here
        canvas = canvases[p] # [3, 128, 128]
        stroke = strokes[p].unsqueeze(0) # [1, 13]

        if level >= start_organic_level and args.salient_mask:
            #idx_nopaint = onehot[p].item() # index to not paint
            if p in onehot:
                print('Not painting this stroke')
                invalid_strokes += 1
                continue

            # Checking whether the stroke is outside the mask by adding alphas
            mask_patch = mask_patches[p] # [1, 128, 128]
            stroke_alpha = _forward_pass_l2p_alpha(stroke)[:,0] # [1, 128, 128]
            stroke_alpha[stroke_alpha >= 0.3] = 1.
            stroke_alpha[stroke_alpha < 0.3] = 0.
            mask_patch[mask_patch >= 0.3] = 1.
            mask_patch[mask_patch < 0.3] = 0.
            masks_together = (mask_patch + stroke_alpha).clip(max=1.0)

            if masks_together.sum().item() > mask_patch.sum().item(): # if the number of 1s is higher in mask + stroke than in mask alone, then the stroke goes outside the mask 
                print('Not painting this stroke')
                invalid_strokes += 1
                continue

        # Not using this method
        # if onehot != None: # dont paint if outside mask
        #     paint = onehot[p]
        #     if paint == 0:
        #         print(f'Not painting this stroke onehot: {stroke} ')
        #         continue
        #
        # if torch.sum(stroke[:, 8:10], dim=1) == 0.:
        #     print(f'Not painting this stroke: {stroke} ')
        #     continue

        if args.renderer_type == 'snp':
            foreground, alpha = utils._draw_oilpaintbrush(painter, stroke.squeeze().detach().cpu()) # [128, 128, 3]
            foreground = torch.from_numpy(foreground).permute(2,0,1).to(device) # black background
            alpha = torch.from_numpy(alpha).permute(2,0,1).to(device) # black background # [3, 128, 128]

        else:
            foreground, alpha = utils.texturizer(painter, writer, stroke, 128, p, t)

        canvas = canvas * (1 - alpha) + (foreground * alpha)  # [3, 128, 128]

        writer.add_image(f'stroke_text', img_tensor=foreground, global_step=p)
        writer.add_image(f'alpha_text', img_tensor=alpha, global_step=p)

        if args.video:
            painter.texturized_strokes_level[p, t] = canvas

        canvases[p] = canvas
        alphas.append(alpha)
        foregrounds.append(foreground)
        #self.all_texturized_strokes.append(canvas)
        valid_strokes += 1

    print(f'Invalid strokes in level {level}: {invalid_strokes}')
    print(f'Valid strokes in level {level}: {valid_strokes}')

    return canvases, foregrounds, alphas

def _forwad_pass_l2p(stroke, canvas, brush_size, num_params, renderer, idx=None):
    """
    :param stroke: [n_patches, 13]
    :param canvas: [n_patches, 3, 128, 128]
    :param brush_size:
    :param idx: [n_patches] indexes that should not be painted == make them transparent 
    :return:
    """
    width = canvas.shape[2]
    #print('stroke shape: ', stroke.shape)
    #stroke = stroke.view(-1, self.num_params)  # [n_patches, 13]
    stroke = utils.remove_transparency(stroke, num_params)
    
    # Make transparent strokes we don't want to paint
    if idx != None:
        stroke = utils.make_transparent(stroke, idx)

    stroke = utils.clip_width(stroke, num_params, max=brush_size)

    if num_params == 13:
        abrgb = 10
    else:
        abrgb = 14
    
    alpha = 1 - renderer(stroke[:, :abrgb])  # 1 - renderer means changing from white to black background -> INVERTING BITMAP (self.renderer outputs black and white strokes)
    alpha = alpha.view(-1, width, width, 1)  # [N, 128, 128, 1]
    color_stroke = alpha * stroke[:, -3:].view(-1, 1, 1, 3)  # [N, 128, 128, 3]
    alpha = alpha.permute(0, 3, 1, 2)  # [N, 1, 128, 128]
    color_stroke = color_stroke.permute(0, 3, 1, 2)  # [N, 3, 128, 128]
    alpha = alpha.view(-1, 1, width, width)  # [N, 1, 128, 128]
    color_stroke = color_stroke.view(-1, 3, width, width)  # [N, 3, 128, 128]

    # Blend strokes one by one in order
    canvas = canvas * (1 - alpha) + color_stroke  # blending in with canvas [N, 3, 128, 128]

    return canvas, color_stroke, alpha # canvas 'n color stroke [npatches, 3, 128, 128] alpha [npatches, 1, 128,128]

def _forward_pass_snp(stroke, canvas, brush_size, num_params, renderer, idx=None):
    """
    :param stroke:  [N, 12]
    :param canvas: 
    :param brush_size: 
    :param idx: 
    :return: 
    """
    stroke = utils.clip_width(stroke, num_params, max=brush_size)
    v = torch.reshape(stroke, [stroke.shape[0], 12, 1, 1]) # [N, 12, 1, 1]
    G_pred_foregrounds, G_pred_alphas = renderer(v) # [N, 3, 128, 128], [N, 3, 128, 128]
    G_pred_foregrounds = morphology.Dilation2d(m=1)(G_pred_foregrounds) # [N, 3, 128, 128]
    G_pred_alphas = morphology.Erosion2d(m=1)(G_pred_alphas) # [N, 3, 128, 128]

    G_pred_foregrounds = G_pred_foregrounds.clip(min=0, max=1)
    G_pred_alphas = G_pred_alphas.clip(min=0, max=1)

    canvas = canvas * (1 - G_pred_alphas) + (G_pred_foregrounds * G_pred_alphas)
    return canvas, G_pred_foregrounds, G_pred_alphas


def blend_general_canvas(canvas, general_canvas, args, patches_limits, npatchesW):
    """
    :canvas: [N, 3, H, W] -> N is number of patches 
    :general canvas:  [1, 3, H, W]
    :param stroke: [n_patches, 13]
    :param canvas: [n_patches, 3, 128, 128]
    :param brush_size: 
    :return: 
    """
    ov = args.overlap
    
    # iterate over all patches and paste them patch by patch
    for i in range(len(patches_limits)):
        curr_patch_loc = patches_limits[i] # tuple of (start_h, end_h)(start_w,end_w)
        h_st = curr_patch_loc[0][0]
        h_end = curr_patch_loc[0][1]
        w_st = curr_patch_loc[1][0]
        w_end = curr_patch_loc[1][1]

        general_canvas[:,:,h_st:h_end,w_st:w_end] = canvas[i]

        #Fix edges: We have the same number of hard edges than patches: if 4 patches, 4 edges to fix, etc.
        if i > 0 and ov > 0:
            
            prev_patch_loc = patches_limits[i-1]  # tuple of (start_h, end_h)(start_w,end_w)
            prev_h_st = prev_patch_loc[0][0]
            prev_h_end = prev_patch_loc[0][1]
            prev_w_st = prev_patch_loc[1][0]
            prev_w_end = prev_patch_loc[1][1]

            # Vertical overlaps : Avoids doing the overlap in the leftmost patches
            if i % npatchesW != 0:
                # We are fading in the patch that we put on top, gradually.
                t = torch.linspace(0, 1, steps=ov).to(device) # [0,0.1,0.2....1]
                
                # Smooth transition between the 2 adjacent canvas (blends pixels from the 2 adjacent patches)
                patch = (canvas[i-1,:,:,-ov:] * (1-t)) + (canvas[i, :, :, 0:ov] * t) # [3, 128, 20]

                general_canvas[:, :, h_st:prev_h_end, w_st:prev_w_end] = patch.unsqueeze(0)

            # Horizontal overlaps: avoids doing this in the first row
            if i >= npatchesW: # horizontal edge as well
                t = torch.linspace(0, 1, steps=ov).to(device)  # [0,0.1,0.2....1]
                # We need to permute to multiply by t
                patch = (canvas[i, :, 0:ov, :].permute(0,2,1) * t) + (canvas[i-npatchesW, :, -ov:, :].permute(
                    0,2,1) * (1-t))
                patch = patch.permute(0,2,1)

                general_canvas[:, :, h_st:h_st + ov, w_st:w_st + 128] = patch.unsqueeze(0)

    return general_canvas


def blend_general_canvas_natural(canvas, patches_limits, general_canvas, blendin=False):
    """
    :general canvas: is [1, 3, H, W]
    :param canvas: [n_patches, 3, 128, 128]
    :return:
    """        
    kanvas = general_canvas # to shorten the name 
    
    # iterate over all patches and paste them patch by patch
    for i in range(len(patches_limits)):
        curr_patch_loc = patches_limits[i] # tuple of (start_h, end_h)(start_w,end_w)
        curr_canvas = canvas[i]

        h_st = curr_patch_loc[0][0]
        h_end = curr_patch_loc[0][1]
        w_st = curr_patch_loc[1][0]
        w_end = curr_patch_loc[1][1]

        # Blends the perimeter of each patch with the general canvas. 
        if blendin:
            ov = 20
            t = torch.linspace(0, 1, steps=ov).to(device)
            
            # Right side of patch
            patch = (kanvas[:, :, h_st:h_end, w_end-ov:w_end] * t) + \
                    (curr_canvas[:, :, -ov:] * (1 - t))  # [3, 128, 20]
            kanvas[:, :, h_st:h_end, w_end-ov:w_end] = patch

            # Left side of patch
            patch = (kanvas[:, :, h_st:h_end, w_st:w_st+ov] * (1 - t)) + \
                            (curr_canvas[:, :, 0:ov] * t)  # [3, 128, 20]
            kanvas[:, :, h_st:h_end, w_st:w_st+ov] = patch

            # Top side - I think the permute is because of the blending left to right 
            patch = (kanvas[:, :, h_st:h_st+ov, w_st+ov:w_end-ov].permute(0,1,3,2) * (1 - t)) + \
                            (curr_canvas[:, 0:ov, ov:-ov].permute(0,2,1) * t)  # [3, 128, 20]
            patch = patch.permute(0,1,3,2)
            kanvas[:, :, h_st:h_st+ov, w_st+ov:w_end-ov] = patch

            # Bottom side
            patch = (kanvas[:, :, h_end-ov:h_end, w_st+ov:w_end-ov].permute(0,1,3,2) * t) + \
                            (curr_canvas[:, -ov:, ov:-ov].permute(0,2,1) * (1 - t))  # [3, 128, 20]
            patch = patch.permute(0,1,3,2)
            kanvas[:, :, h_end - ov:h_end, w_st + ov:w_end - ov] = patch

        # Directly pastes the center of the patch onto the general canvas (does not blend it)
        kanvas[:,:, h_st+ov:h_end-ov, w_st+ov:w_end-ov] = curr_canvas[:, ov:128-ov, ov:128-ov]

    # Visualize and write in logger
    #writer.add_image(f'general_canvas_texture_{texture}', img_tensor=kanvas.squeeze(0), global_step=0)

    return kanvas 


def blend_diff(crops, patches_limits, general_canvas, alpha=1.0):
    N = len(crops)
    blend = general_canvas.clone().detach().requires_grad_() # [1, 3, H, W]
    
    for i in range(N):
        curr_patch_loc = patches_limits[i] # tuple of (start_h, end_h)(start_w,end_w)
        crop = crops[i].unsqueeze(0) # [1, 3, 128, 128]
        
        h_st = curr_patch_loc[0][0]
        h_end = curr_patch_loc[0][1]
        w_st = curr_patch_loc[1][0]
        w_end = curr_patch_loc[1][1]

        # blended = blend[:, :, h_st:h_end, w_st:w_end] + crop
        # blend[:, :, h_st:h_end, w_st:w_end] = blended

        # Generate mask with the general canvas size 
        mask = torch.zeros_like(blend).to(device)
        mask[:, :, h_st:h_end, w_st:w_end] = 1 # assign ones where the region is 

        blend_image = torch.where(mask > 0, crop, blend)

    return blend

def blend_diff_v1(crops, patches_limits, general_canvas, alpha=1.0):
    batch_size = 1
    crop_height, crop_width = 128, 128
    for i in range(len(patches_limits)):
        # Create mask 
        curr_patch_loc = patches_limits[i] # tuple of (start_h, end_h)(start_w,end_w)
        crop = crops[i]

        h_st = curr_patch_loc[0][0]
        h_end = curr_patch_loc[0][1]
        w_st = curr_patch_loc[1][0]
        w_end = curr_patch_loc[1][1]

        # Get regions from the bigger image we want to paste  
        blend_region = general_canvas[:, :, h_st:h_end, w_st:w_end]
        
        grid = torch.zeros(batch_size, 2, crop_height, crop_width, device=device) # [N, 2, crop_height, crop_width]
        grid[:, 0, :, :] = torch.linspace(w_st, w_end, crop_width).repeat(batch_size, crop_height, 1).to(device)
        grid[:, 1, :, :] = torch.linspace(h_st, h_end, crop_height).repeat(batch_size, crop_width, 1).transpose(1, 2).to(device)
        grid = grid.to(torch.float64).permute(0,2,3,1) / 128.

        blend_image = F.grid_sample(general_canvas, grid, mode='nearest', padding_mode='zeros')

        # create a binary mask indicating the blend region
        mask = torch.zeros_like(blend_region)
        mask[:, :, crop_height//2:-crop_height//2, crop_width//2:-crop_width//2] = 1
        
        # blend the crop into the blend region using a differentiable operation
        blend_region = blend_region * (1 - crop.squeeze(0)) + crop.squeeze(0)
        blend_image = torch.where(mask > 0, blend_region, general_canvas)

        #blend_image[:, :, h_st:h_end, w_st:w_end] = blend_image[:, :, h_st:h_end, w_st:w_end] * (1 - crop.squeeze(0)) + crop.squeeze(0)
        #pdb.set_trace()
    return blend_image 

def blend_general_canvas_gif(self, canvas, general_canvas, patch_limit):
    """
    general canvas is [1, 3, H, W]
    :param canvas: [3, 128, 128]
    :param brush_size:
    :return:
    """
    # tuple of (start_h, end_h)(start_w,end_w)
    h_st = patch_limit[0][0]
    h_end = patch_limit[0][1]
    w_st = patch_limit[1][0]
    w_end = patch_limit[1][1]
    general_canvas[:,:,h_st:h_end,w_st:w_end] = canvas
    # general_canvas[:,:, h_st+ov:h_end-ov, w_st+ov:w_end-ov] = canvas[:, ov:128-ov, ov:128-ov]
    return general_canvas

def blend_general_canvas_organic_gif(self, canvas, general_canvas, patch_limit):
    """
    general canvas is [1, 3, H, W]
    :param canvas: [3, 128, 128]
    :return:
    """
    h_st = patch_limit[0][0]
    h_end = patch_limit[0][1]
    w_st = patch_limit[1][0]
    w_end = patch_limit[1][1]
    ov = 20
    general_canvas[:,:, h_st+ov:h_end-ov, w_st+ov:w_end-ov] = canvas[:, ov:128-ov, ov:128-ov]

    return general_canvas


