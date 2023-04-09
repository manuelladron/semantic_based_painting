import numpy as np 
import torch 
from utils import utils 
import pdb 
import torch.nn.functional as F
#import morphology
import kornia 
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def _forward_pass_l2p_alpha(action, num_params, renderer, use_transp, i_til_rgb=10):
    if use_transp == False:
        action = utils.remove_transparency(action, num_params)
    stroke = 1 - renderer(action[:, :i_til_rgb]).unsqueeze(1) # [1, 128, 128] -> [1, 1, 128, 128]
    return stroke.repeat(1,3,1,1) # returns alpha with 3 channels

def texturize(strokes, canvases, brush_size, t, num_params, writer, level, onehot=None, mask_patches=None, painter=None, segm_name='', use_transp=True):
    """
    Processes textures stroke by stroke

    Args:
        strokes: [n_patches, 13], t-th stroke in every patch 
        canvases: [n_patches, 3, 128, 128]
        brush_size: float to cap max brushsize
        t: index in the stroke budget 
        num_params: int, 13 if bezier 

        :return: texturized canvases 
    """
    valid_strokes = 0
    invalid_strokes = 0
    if use_transp == False:
        strokes = utils.remove_transparency(strokes, num_params=num_params)
    strokes = utils.clip_width(strokes, num_params, max=brush_size)
    
    # iterate over patches and apply texture one by one
    alphas, foregrounds = [], []
    npatches = strokes.shape[0]
    
    for p in range(npatches): # Iterate over patches. Single stroke here
        canvas = canvases[p]             # [3, 128, 128]
        stroke = strokes[p].unsqueeze(0) # [1, 13]

        # Get texturized stroke
        foreground, alpha = texturizer(painter, writer, stroke, 128, p, t, level, segm_name=segm_name)

        # Update canvas patch 
        canvas = canvas * (1 - alpha) + (foreground * alpha)  # [3, 128, 128]

        writer.add_image(f'stroke_text', img_tensor=foreground, global_step=p)
        writer.add_image(f'alpha_text', img_tensor=alpha, global_step=p)

        canvases[p] = canvas
        alphas.append(alpha)
        foregrounds.append(foreground)
        #self.all_texturized_strokes.append(canvas)
        # valid_strokes += 1

    # print(f'Invalid strokes in level {level}: {invalid_strokes}')
    # print(f'Valid strokes in level {level}: {valid_strokes}')

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
    kanvas = general_canvas 
    
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

def calculate_euclidean_dist(c0, c1):
    # c0, c1 -> [1, 2]
    # print(f'first point: {c0}, second point: {c1}')
    sum_sq = torch.sum(torch.square(c0 - c1), dim=1)
    euclidean_dist = torch.sqrt(sum_sq)
    return euclidean_dist

def texturizer(painter, writer, stroke_param, canvas_size, patch, time, level, segm_name=''):
    """
    Apply texture stroke by stroke. Depending on the shape of the stroke, will apply texture in different ways
    :param painter: 
    :param writer: 
    :param stroke_param: [1,13]
    :param canvas_size:  128
    :param brush_size:   varies
    :return: 
    """
    #print(f'segm_name: {segm_name}')
    if segm_name == 'sky' or level == 0:
        t1 = '../../painting_tools/brushes/noshape_light_more.png'
    else:
        t1 = '../../painting_tools/brushes/noshape_light.png'
    
    # t1 = '../painting_tools/brushes/brush_large_vertical_clean2.png'
    # t2 = '../painting_tools/brushes/acrylic_w2.png'
    # t3 = '../painting_tools/brushes/oil_bump.png'

    # id_text = np.random.randint(0, 2)
    # textures = [t1, t3]
    # chosen_texture = textures[id_text]

    # Extract coordinates 
    y0, x0 = stroke_param[0, :2]
    y1, x1 = stroke_param[0, 2:4]
    y2, x2 = stroke_param[0, 4:6]

    y1 = y0 + (y2 - y0) * y1
    x1 = x0 + (x2 - x0) * x1

    c0 = stroke_param[:, :2]
    c1 = torch.stack((y1, x1)).unsqueeze(0).to(device)
    c2 = stroke_param[:, 4:6]

    # Stroke length 
    ed1 = calculate_euclidean_dist(c0, c1)
    ed2 = calculate_euclidean_dist(c1, c2)
    stroke_length = ed1 + ed2

    # print('stroke lenght in pixels: ', stroke_length * canvas_size)
    # print('stroke lenght normalized: ', stroke_length)
    # print('firsthalf lenght normalized: ', ed1)
    # print('secondhalf lenght normalized: ', ed2)

    avg_rad = (stroke_param[0, 6] + stroke_param[0, 7]) / 2
    # print('radius average: ', avg_rad)

    ratio = stroke_length / avg_rad
    # print(f'ratio length/radius: {ratio}')
    linear = 0
    curved = 0

    # pdb.set_trace()
    # if avg_rad > 0.8 or ratio < 2 or stroke_length < 0.1 or ed1 < 0.15 or ed2 < 0.15:
    linear += 1
    x_ct = (x0 + x2) / 2
    y_ct = (y0 + y2) / 2
    ct_pt = torch.stack((y_ct, x_ct)).unsqueeze(0).to(device)
    fore, alpha = add_linear_texture(painter, stroke_param, canvas_size, t1, ct_pt, stroke_length, visualize=False)
        # writer.add_image(f'text_linear_p_{patch}_t_{time}', img_tensor=fore,global_step=0)
        # pdb.set_trace()
    # else:
    #     curved += 1
    #     fore, alpha = add_curved_texture(painter, stroke_param, canvas_size, t1, postprocess=True)
        # writer.add_image(f'text_curved_p_{patch}_t_{time}', img_tensor=fore, global_step=0)
        # pdb.set_trace()

    fore = fore.clip(min=0, max=1)
    alpha = alpha.clip(min=0, max=1)
    # print('in texturizer')
    # pdb.set_trace()
    return fore, alpha  # [3, 128, 128]


def add_linear_texture(painter, stroke_param, canvas_size, texture_path, center_pt, stroke_length, visualize=False):
    """
    Function that rotates and translates bitmap according to stroke parameters.
    Center point is [1, 2], and it is the target of translation.
    """
    # Read texture bitmap 
    img = cv2.imread(texture_path, cv2.IMREAD_COLOR)[:, :, ::-1]
    img = cv2.resize(img, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)  # [128,128,3] uint8 [0-255]

    if visualize:
        plt.imshow(img)
        plt.show()

    # Convert to pytorch
    img_p = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(dim=0).to(device) /255 # [1, 3, 128, 128] uint8 [0-255]

    # If stroke is too long, we rescale a bit the bitmap 
    # if stroke_length.item() > 0.65:
    #     scale_factor = torch.FloatTensor([1, 2]).unsqueeze(0).to(device)
    #     img_p = kornia.geometry.transform.scale(img_p / 255, scale_factor)
    #     if visualize:
    #         sc_np = img_p * 255
    #         sc_np = sc_np.squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    #         plt.imshow(sc_np)
    #         plt.show()

    ##### 1) ROTATION over brush stroke center X,Y = 70, 60 # Calculated by eye from bitmap
    X, Y = 64, 64 #70, 60  # Calculated by eye from bitmap
    brush_center = torch.Tensor((X, Y)).unsqueeze(0).to(device)
    # Find angle between start and end points
    y0, x0 = stroke_param[0, :2]  # [2]
    y2, x2 = stroke_param[0, 4:6]  # [2]

    # https://stackoverflow.com/questions/42258637/how-to-know-the-angle-between-two-vectors
    theta_radians = torch.atan2(-(y2 - y0), (x2 - x0))
    theta_degrees = torch.rad2deg(theta_radians) + 90

    # if stroke_length.item() <= 0.65:
    #     img_p = img_p / 255
    rotation = kornia.geometry.transform.rotate(img_p, theta_degrees, center=brush_center, padding_mode='reflection')
    # [0-1] range

    # Visualization
    if visualize:
        rot_np = rotation * 255
        rot_np = rot_np.squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        plt.imshow(rot_np)
        plt.show()

    transl = rotation
    ##### 2) TRANSLATION: update (05.11.22) No need to translate
    # y, x = center_pt.squeeze()
    # center_pt_xy = torch.Tensor([x, y]).unsqueeze(0).to(device)
    # center_pt_xy = center_pt_xy * canvas_size  # [1, 2]
    # translation_vector = center_pt_xy - brush_center
    #
    # transl = kornia.geometry.transform.translate(rotation, translation_vector, mode='bilinear', padding_mode='zeros',
    #                                              align_corners=True)  # [0-1] range
    # pdb.set_trace()
    # if visualize:
    #     transl_np = transl * 255
    #     transl_np = transl_np.squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    #     plt.imshow(transl_np)
    #     plt.show()

    # Mask out the shape and add color
    i_til_rgb = 10 if painter.num_params == 13 else 8
    alpha = _forward_pass_l2p_alpha(stroke_param, painter.num_params, painter.renderer, painter.args.use_transparency, i_til_rgb)  # [1,3,128,128] [0-1] range 

    if visualize:
        alpha_np = (alpha.squeeze() * 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)  # [H,W,3]
        plt.imshow(alpha_np)
        plt.show()

    # Reshapes texture into alpha shape : basically, crops the bitmap with the alpha shape of the stroke 
    transl[alpha <= 0.5] = 0  # [0-1] range

    # text_alpha_np = (transl*255).squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)  # [H,W,3]
    text_alpha_np = (transl).squeeze().permute(1, 2, 0).detach().cpu().numpy()  # [H,W,3]
    text_alpha = text_alpha_np.copy()

    # Set color 
    if painter.num_params == 13:
        R, G, B = stroke_param[0, 10], stroke_param[0, 11], stroke_param[0, 12]
    
    else:
        R, G, B = stroke_param[0, 8], stroke_param[0, 9], stroke_param[0, 10]
    
    text_alpha_np[:, :, 0] = text_alpha_np[:, :, 0] * R.item()  # [3, 128, 128]
    text_alpha_np[:, :, 1] = text_alpha_np[:, :, 1] * G.item()  # text alpha is in [0-255], multiplied by 0.4 gives us
    text_alpha_np[:, :, 2] = text_alpha_np[:, :, 2] * B.item()

    foreground = cv2.dilate(text_alpha_np, np.ones([2, 2]))
    alpha = cv2.erode(text_alpha, np.ones([2, 2]))

    if visualize:
        # plt.imshow(alpha)
        # plt.show()
        plt.imshow(foreground)
        plt.show()

    foreground = torch.FloatTensor(foreground).permute(2, 0, 1).to(device)  # / 255
    alpha_text = torch.FloatTensor(text_alpha).permute(2, 0, 1).to(device)  # / 255

    # alpha_text[alpha_text > 0.5] = 1 # 0.3
    # print('in linear texture')
    # pdb.set_trace()
    return foreground, alpha_text  # alpha.squeeze()