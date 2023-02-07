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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# IMAGE PROCESING -------------- 
def resize_tensor(tensor, new_h, new_w):
    # Define the desired output size
    output_size = (new_h, new_w)

    # Resize the tensor using bilinear interpolation
    resized_tensor = torch.nn.functional.interpolate(tensor.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False).squeeze()

    return resized_tensor


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def normalize_img(img):
    """Converts range 0-255 to 0-1"""
    return (img - np.min(img)) / np.ptp(img)
    
def init_logger(args):

    dirname = args.save_dir
    subdir = 'logs'
    name = args.exp_name

    fullpath = os.path.join(dirname, subdir, name)
    logger = SummaryWriter(fullpath)
    print(f'logger created in {fullpath}')
    return logger 

def resize_based_on_patches(src_img, args, h, w, mask=None):
    """
    Resizes image based on 128x128 patches 
    """
    
    # Calculate number of patches per side 
    npatches_h = int(np.ceil(h / 128))
    npatches_w = int(np.ceil(w / 128))
    
    sizeH = (npatches_h * 128) - (args.overlap * (npatches_h - 1))
    sizeW = (npatches_w * 128) - (args.overlap * (npatches_w - 1))
    
    sizeH = int(sizeH)
    sizeW = int(sizeW)

    src_img = cv2.resize(src_img, (sizeW, sizeH))

    mask = 0
    if mask:
        mask = cv2.resize(mask, (sizeW, sizeH))
    
    return src_img, npatches_h, npatches_w, mask

def resize_segmentation_mask(mask, height, width):
    # Convert the NumPy array to a PIL image
    mask_image = Image.fromarray(np.uint8(mask))
    # Resize the image to the specified height and width
    resized_mask = mask_image.resize((width, height), resample=Image.NEAREST)
    # Convert the resized image back to a NumPy array
    resized_mask = np.array(resized_mask)
    return resized_mask

def map_segmentation_mask_to_colors(mask):
    categories_json = '/home/manuelladron/projects/npp/stroke_opt_main/stroke_opt_23/utils/coco_panoptic_cat.json'
    
    with open(categories_json) as f:
        categories = json.load(f)
    # Create a dictionary mapping the id field of each category to its color 
    id_to_color = {c['id']: c['color'] for c in categories}
    
    # Find all unique ids in the input mask 
    unique_ids = np.unique(mask) 

    # Create a color map by looking up the color for each unique id 
    color_map = np.array([id_to_color[id] for id in unique_ids])
    
    # Initialize a 3-channel color mask to all zeros
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # For each id and its corresponding color, set the pixels in the mask with that id to the corresponding color in the color_mask
    for id, color in zip(unique_ids, color_map):
        color_mask[mask == id] = color
    
    return color_mask


def overlay_image_and_mask(image, color_mask, alpha=0.5):
    # Normalize the image and color mask arrays to values between 0 and 1
    image = image / 255.0
    color_mask = color_mask / 255.0
    
    # Create a composite image by combining the original image and the color mask using the specified alpha value
    composite_image = image * (1.0 - alpha) + color_mask * alpha
    
    # Clip the composite image to the range [0, 1] to avoid overflow or underflow
    composite_image = np.clip(composite_image, 0.0, 1.0)
    
    # Convert the composite image back to 8-bit integers and return it
    return (composite_image * 255).astype(np.uint8)

def process_img(args, img_path, writer, resize_value=128, min_width=400):
    """
    Receives image path, opens and resizes it and returns tensor 
    """

    # Open image and resize it 
    src_img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:,:,::-1] # from BGR to RGB uint8 [H,W,3]
    print(f'Original image size H={src_img.shape[0]} x W={src_img.shape[1]}')
    
    # If passing a resizing value 
    if resize_value != None:
        src_img = cv2.resize(src_img, (resize_value, resize_value)) # [0-255]
        new_h, new_w = resize_value
    
    else: 
        # Decrease size if it's too big 
        if src_img.shape[0] or src_img.shape[1] > 1200:
            new_h = src_img.shape[0] // args.aspect_ratio_downsample
            new_w = src_img.shape[1] // args.aspect_ratio_downsample
        
        elif args.upsample and (new_h < min_width or new_w < min_width): # Adjust it to be at least over 1000 pixels so the painting is not too small 
            src_img = image_resize(src_img, width=min_width)
            
            if args.salient_mask != '':
                mask = image_resize(args.salient_mask, width=min_width)
            
            new_h = src_img.shape[0]
            new_w = src_img.shape[1]
        
        else:
            new_h = src_img.shape[0] 
            new_w = src_img.shape[1] 
        
        # Temporarily turning this off 
        # while new_h > 1300 or new_w > 1300:
        #     args.aspect_ratio_downsample += 0.1
        #     new_h = src_img.shape[0] // args.aspect_ratio_downsample
        #     new_w = src_img.shape[1] // args.aspect_ratio_downsample
            
        #     print(f'Image larger than 1400 pixels in one of the dimensions, downsampling it a bit')
        #     print(f'downsampled H: {self.original_H}, downsampled W : {self.original_W}')

    npatches_h, npatches_w, mask = 0, 0, None
    if args.paint_by_patches:
        src_img, npatches_h, npatches_w, mask = resize_based_on_patches(src_img, args, new_h, new_w, mask=None) # src img is [H, W, 3]
    
    # Calculate segmentation map 
    segm_ids, boundaries = 0, 0
    if args.use_segmentation_mask:
        # Option 1 -> Calculate segmentation mask based on original image size, then resize 
        segm_ids = segment_image(img_path, is_path=True) # [H, W, 3], [H, W]
        segm_ids = resize_segmentation_mask(segm_ids, src_img.shape[0], src_img.shape[1])

        # Find boundaries 
        boundaries = find_boundaries(segm_ids, mode='thick') # [H, W]
        boundaries = np.expand_dims(boundaries, axis=0) # [1, H, W]
        boundaries = torch.from_numpy(boundaries) # [1, H, W] -> Boolean tensor with boundaries selected
        
        segm_mask_color = map_segmentation_mask_to_colors(segm_ids) # [H, W, 3]

        # Overlay for visualization purposes 
        segm_mask_overlay = overlay_image_and_mask(src_img, segm_mask_color, alpha=0.5)
        segm_mask_overlay = torch.from_numpy(segm_mask_overlay.transpose(2,0,1)) # [C, H, W]

        # segmentation mask 
        segm_mask_color = normalize_img(segm_mask_color)
        segm_mask_color = torch.from_numpy(segm_mask_color.transpose(2,0,1)) # [C, H, W]
        
        # Depict boundaries
        segm_mask_color_with_boundaries = torch.where(boundaries == True, 0.0, segm_mask_color)

        writer.add_image('segm_original', img_tensor=segm_mask_color_with_boundaries, global_step=0)
        writer.add_image('segm_original_overlay', img_tensor=segm_mask_overlay, global_step=0)

    # Normalize it and conver to torch tensor 
    src_img = normalize_img(src_img)
    img = torch.from_numpy(src_img.transpose(2,0,1)).unsqueeze(0) # [1, C, H, W]
    
    print(f'Adjusted input image -> H={img.shape[2]}, W={img.shape[3]}')
    return img, mask, npatches_h, npatches_w, segm_ids, boundaries


def increase_boundary_thickness(binary_image, kernel_size=3, stride=1, padding=0):
    # Expand the binary image to 3D, with a single channel and height/width equal to the original size
    binary_image = binary_image.unsqueeze(0).unsqueeze(0)

    # Perform max pooling with the specified kernel size, stride, and padding
    dilated_image = F.max_pool2d(binary_image, kernel_size, stride, padding)

    # Squeeze the result back to 2D
    dilated_image = dilated_image.squeeze().squeeze()

    return dilated_image


# PATCHING UTILITIES -------------- 
def merge_tensors(tensor_a, tensor_b, indices_a, indices_b):
    """
    Merges two tensors given a list of indices corresponding to tensor_a and tensor_b

    :param tensor_a: batch tensor with smaller number of tensors
    :param tensor_b: batch tensor containing all tensors 

    :returns: merged tensor 
    """
    N, C, H, W = tensor_a.shape # N is a subset of M
    M, _, _, _ = tensor_b.shape # M is all indices 

    # Create new tensor with all M patches 
    merged_tensor = torch.zeros(M, C, H, W).to(device)
    
    # Double-check N and M correspond to the lengths of indices_a and indices_b respectively
    assert N == len(indices_a)
    assert (M - N) == len(indices_b)
    
    for i in range(N): # 0, 1, 2....
        # indices_a[i] = 37, 68, 3, 15, ...
        #print(f'i: {i}, indices_a[i]: {indices_a[i]}')
        merged_tensor[indices_a[i]] = tensor_a[i]

    for j in range(M-N):
        #print(f'j: {j}, indices_b[j]: {indices_b[j]}')
        merged_tensor[indices_b[j]] = tensor_b[indices_b[j]]

    return merged_tensor 


def select_tensors_with_n_true(tensor_list, limits_list, N):
    """
    Selects indexes of tensors that at least have >= N true/1s values 
    Returns list with selected indices and total number of selected tensors 
    """
    selected_tensors_idx = []
    selected_tensors = []
    n_total = 0
    for i, tensor in enumerate(tensor_list):
        if (tensor.sum().item() >= N):
            selected_tensors_idx.append(i)
            selected_tensors.append(tensor)
            n_total += 1

    print(f'\nNumber of selected tensors that have more than {N} border pixels: {n_total}')
    return selected_tensors_idx, selected_tensors, n_total


def get_patches_w_overlap(args, img, npatchesH, npatchesW, writer=None, name='src_img'):
    """
    Scan an image with a step size and stores its corresponding patches
    :param img: general size image of shape [1, 3, H, W]
    :param mask: either image of same shape as img [1, 3, H, W] or integer 0 
    :return: list with patches, and corresponding list with the patches boundaries in the format ((h_st, h_end), (w_st, w_end))
    """
    patches = []
    patches_limits = []  # Also for mask 
    
    for h in range(npatchesH): 
        start_h = max((128 * h) - (args.overlap*h), 0) # so it doesn't get negative numbers
        
        for w in range(npatchesW):
            start_w = max((128 * w) - (args.overlap*w), 0) # so it doesn't get negative numbers
            end_w = 128 + (w * (128 - args.overlap))
            end_h = 128 + (h * (128 - args.overlap))
            
            patch = img[:, :, start_h:end_h, start_w:end_w]
            patches.append(patch)
            patches_limits.append([(start_h,end_h),(start_w,end_w)])

    if writer != None:
        img_grid = torchvision.utils.make_grid(torch.cat(patches,dim=0), nrow=npatchesW)
        writer.add_image(f'{name}_by_patches', img_tensor=img_grid, global_step=0)
    
    return patches, patches_limits

def crop_image(patches_loc, image, return_tensor=True):
    """Given a list of patches boundaries and an image, crops image.
    :return: a concatenated list of patches"""
    patches = []
    for i in range(len(patches_loc)):
        (start_h, end_h), (start_w, end_w) = patches_loc[i]
        patch = image[:, :, start_h:end_h, start_w:end_w]
        patches.append(patch)
    
    if return_tensor:
        patches_image = torch.cat(patches, dim=0)
        return patches_image
    
    return patches

def is_tensor(var):
    return isinstance(var, torch.Tensor)

def create_N_random_patches(image, N, mask=None, threshold=60, patch_size=128):
    """
    Creates a list of N random patches boundaries. Does not return patches itself, just the list of boundaries  
    
    :param image: Reference image of shape [1, 3, H, W]
    :param N: number of patches to create, an integer 
    :param threshold: max amount of pixels that are allowed to overlap between patches
    
    :return: a list of the N random patches limits 
    """

    H = image.shape[2]
    W = image.shape[3]

    patches_limits = []
    i = 0 # iterations, to check the proportion of successful patches 
    d = 0 # number of ditched patches
    
    if is_tensor(mask):
        mask = mask.clone() # this copy will be updated so that we avoid patch repetition

    # Loop to keep adding patches until we reach the number we want 
    while len(patches_limits) != N: 
        
        # Create patches from upper left coordinate 
        xcoord = int(torch.randint(W - patch_size, ()))
        ycoord = int(torch.randint(H - patch_size, ()))

        # Coordinates of the random patch 
        start_w = xcoord
        end_w = xcoord + patch_size
        start_h = ycoord
        end_h = ycoord + patch_size
        
        # This approach gets patches based on whether center point of random patch falls within the mask
        if is_tensor(mask):
            
            # Get center coordinates of the patch, and check if center point falls inside the mask. If not, do not add it. 
            cx = (start_w + end_w) // 2 # center x coordinate in patch 
            cy = (start_h + end_h) // 2 # center y coordinate in patch 

            # 1 or 0 if patch falls in mask or not 
            inmask = mask[:,:,cy,cx] # this is either 0 or 1 (inside mask or outside mask)

            # Do not add proposed patch if outside mask
            if inmask == 0:
                i += 1
                d += 1
                if i > 500: break # this is just a gate to avoid having an infine tries here 
                # Go back up the loop 
                continue 

        # Check for overlaps: iterate over the existing list of patches coordinates and check if they overlap. If they do for more than the overlap hp, don't include them         
        if i > 0:
            ditch = False
            for j in range(len(patches_limits)): 
                (min_y_prev, max_y_prev), (min_x_prev, max_x_prev) = patches_limits[j]
                
                if abs(start_w - min_x_prev) < threshold and abs(start_h - min_y_prev) < threshold:
                    ditch = True
                    d += 1
                    break # break the loop as soon as there is one patch that overlaps.

            # add one in the counter and continue in the loop 
            if ditch == True:
                i += 1
                if i > 500: break
                else:
                    continue
        
        # Append patches boundaries 
        patches_limits.append([(start_h, end_h), (start_w, end_w)])

        if is_tensor(mask):
            # Update mask so that it does not repeat patches
            mask[:, :, start_h:end_h, start_w:end_w] = 0 

        #print(f'i: {i}, len patches: {len(patches)}')
        i += 1

    print(f'i: {i}, TOTAL NUMBER OF RANDOM PATCHES: {len(patches_limits)}, number of ditched patches: {d}')
    
    return patches_limits


def create_N_random_patches_BU(source_img, N, mask=None, salient_mask=False, threshold=60, patch_size=128):
    """

    :param source_img: Reference image of shape [1, 3, H, W]
    :param N: number of patches to create, an integer 
    :param threshold: max amount of pixels that are allowed to overlap between patches
    :return: physical patches and internally stores patches limits in self.patches_detail_limits
    """

    H = source_img.shape[2]
    W = source_img.shape[3]

    patches_mask = []
    patches = []
    patches_limits = []
    i = 0 # iterations, to check the proportion of successful patches 
    d = 0 # number of ditched patches
    
    if salient_mask:
        mask = mask.clone() # this copy will be updated so that we avoid patch repetition

    # Loop to keep adding patches until we reach the number we want 
    while len(patches) != N: 
        # Create patches from upper left coordinate 
        xcoord = int(torch.randint(W - patch_size, ()))
        ycoord = int(torch.randint(H - patch_size, ()))

        # Coordinates of the random patch 
        start_w = xcoord
        end_w = xcoord + patch_size
        start_h = ycoord
        end_h = ycoord + patch_size
        
        if salient_mask:
            # This approach gets patches based on whether center point of random patch falls within the mask
            
            # Get center coordinates of the patch, and check if center point falls inside the mask. If not, do not add it. 
            cx = (start_w + end_w) // 2 # center x coordinate in patch 
            cy = (start_h + end_h) // 2 # center y coordinate in patch 

            # 1 or 0 if patch falls in mask or not 
            inmask = mask[:,:,cy,cx] # this is either 0 or 1 (inside mask or outside mask)

            # Do not add proposed patch if outside mask
            if inmask == 0:
                i += 1
                d += 1
                if i > 500: break # this is just a gate to avoid having an infine tries here 
                # Go back up the loop 
                continue 

        # Check for overlaps: iterate over the existing list of patches coordinates and check if they overlap. If they do for more than the overlap hp, don't include them         
        if i > 0:
            ditch = False
            for j in range(len(patches_limits)): 
                (min_y_prev, max_y_prev), (min_x_prev, max_x_prev) = patches_limits[j]
                
                if abs(start_w - min_x_prev) < threshold and abs(start_h - min_y_prev) < threshold:
                    ditch = True
                    d += 1
                    break # break the loop as soon as there is one patch that overlaps.

            # add one in the counter and continue in the loop 
            if ditch == True:
                i += 1
                if i > 500: break
                else:
                    continue
        
        # Crop patch and append 
        patch = source_img[:, :, start_h:end_h, start_w:end_w]
        patches.append(patch)
        patches_limits.append([(start_h, end_h), (start_w, end_w)])

        if salient_mask:
            patch_mask = mask[:, :, start_h:end_h, start_w:end_w]
            patches_mask.append(patch_mask)

            # Update mask so that it does not repeat patches
            mask[:, :, start_h:end_h, start_w:end_w] = 0 

        #print(f'i: {i}, len patches: {len(patches)}')
        i += 1

    print(f'i: {i}, TOTAL NUMBER OF RANDOM PATCHES: {len(patches)}, number of ditched patches: {d}')
    
    if salient_mask:
        return patches, patches_limits, patches_mask 
    
    return patches, patches_limits, None


def high_error_candidates(canvas, patches_list, patches_loc_list, level,  num_patches_this_level):
    """
    Resets the number of total patches based on how many quasi non-overlapping patches we can find
    :param canvas: [1, 3, H, W] general canvas
    :param patches_loc_list: list with coordinate patches tuples (start_h, end_h), (start_w, end_w)
    :param patches_list: list with patches of shape [3, 128, 128]
    :return: 
    """

    # 1) crop canvas with patches loc
    # 2) calculate error maps
    # 3) sort and select topk
    
    patches = [] # get patches from canvas 
    for i in range(len(patches_loc_list)):
        (start_h, end_h), (start_w, end_w) = patches_loc_list[i]
        patch = canvas[:, :, start_h:end_h, start_w:end_w]
        patches.append(patch)

    patches_canvas = torch.cat(patches, dim=0)
    patches_img = torch.cat(patches_list, dim=0)
    
    # Calculate l1 error 
    errors = F.l1_loss(patches_canvas, patches_img, reduction='none').mean((1,2,3)) # batch 
    
    # Get number of patches used in this level
    k = num_patches_this_level

    # if number hp patches at this level is higher than the actual amount of patches, limit the number of hp patches 
    if k >= len(errors): 
        k = len(errors) - 1
    total_number_patches = k

    values, indices = torch.topk(errors, k=k)
    selected_patches_loc = [*map(patches_loc_list.__getitem__, indices.tolist())] # list with the selected patches limits (I think it replaces a loop)
    selected_patches_img = torch.index_select(patches_img, 0, indices).to(device) # select patches based on indixes along dimension 0 

    return selected_patches_loc, selected_patches_img, indices, values

def get_natural_patches(number_uniform_patches, source_img, general_canvas, logger, level, mask, salient_mask, boundaries, number_natural_patches, path=None):
    """
    Gets source_image patches with a high-error map algorithm.
    boundaries is a tensor of shape [C, H, W]
    """
    # 1) Get random patches 
    num_patches = int(number_uniform_patches / 1.2) # slightly fewer patches than the amount of uniform patches 

    # Get list of patches limits 
    list_patches_limits = create_N_random_patches(source_img, num_patches, mask=mask)

    # Get source image patches 
    list_natural_src_patches = crop_image(list_patches_limits, source_img, return_tensor=False)
    
    #list_natural_src_patches, list_patches_limits, list_mask_patches = create_N_random_patches(source_img, num_patches, mask=mask, salient_mask=salient_mask)

    # 2) Pseudo-Attention: compute high level error between the list of patches and sort them to paint it by priority 
    print(f'Calculating high error candidates')
    patches_limits, target_patches, indices, values = high_error_candidates(general_canvas, list_natural_src_patches, list_patches_limits, 
                                                                                     level, number_natural_patches)

    mask_patches, boundaries_patches = None, None
    if salient_mask:
        mask_patches = crop_image(list_patches_limits, mask, return_tensor=True)
        mask_patches = torch.index_select(mask_patches, 0, indices) # [k, 1, 128, 128]

    if boundaries != None:
        boundaries_patches = crop_image(list_patches_limits, boundaries, return_tensor=True).to(device)
        boundaries_patches = torch.index_select(boundaries_patches, 0, indices.to(device)) # [k, 1, 128, 128]

    draw_bboxes(general_canvas, patches_limits, level, logger, canvas=True, path=path)
    draw_bboxes(source_img, patches_limits, level, logger, canvas=False, path=path)   # [npatches, 3, 128,128]
    
    return target_patches, patches_limits, indices, values, mask_patches, boundaries_patches 


# STROKES UTILITIES -------------- 

def create_grid_strokes(budget, num_params, device, std = 0.05):
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

    # Other parameters: radius, transparency and rgb 
    rad = torch.rand(budget, 2, requires_grad=True, device=device)
    transp = torch.ones(budget, 2, requires_grad=True, device=device)
    rgb = torch.rand(budget, 3, requires_grad=True, device=device)

    # Put them together 
    strokes = torch.cat([x0,y0, x, y, x2,y2, rad, transp, rgb], dim=1).requires_grad_()

    return strokes 

def init_strokes(budget, mode, device, num_params=13):
    """
    Initializes random strokes parameters on a canvas given a budget
    """
    if mode == 'random':
        # uniform distribution 
        strokes = torch.rand(budget, num_params, requires_grad=True, device=device)
    
    elif mode == 'grid':
        strokes = create_grid_strokes(budget, num_params, device)

    return strokes 

def init_boundary_stroke_params(boundaries, budget, device, std = 0.05):
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

    # Other parameters: radius, transparency and rgb 
    rad = torch.rand(budget, 2, requires_grad=True, device=device)
    transp = torch.ones(budget, 2, requires_grad=True, device=device)
    rgb = torch.rand(budget, 3, requires_grad=True, device=device)

    # Put them together 
    strokes = torch.cat([x0,y0, x, y, x2,y2, rad, transp, rgb], dim=1).requires_grad_()

    return strokes 

def init_coordinates_in_mask(A, N, device):
    """
    Given a boolean tensor A of shape [H, W] and an integer N, returns x,y random coordinates over True values of A. 
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

def init_strokes_patches(budget, mode, device, npatches):
    strokes_l = []
    for p in range(npatches):
        strokes = init_strokes(budget, mode, device)
        strokes_l.append(strokes)
    
    return torch.stack(strokes_l, dim=1).detach().requires_grad_() # [budget, npatches, 13]

def init_strokes_boundaries(N, budget, device, boundaries, patches_limits_list):
    """
    Initializes b-budget strokes (2-4 max) only on patches that have boundaries 
    :param N: number of pixels that represent boundaries.
    :param budget: integer with number of strokes to initialize 
    :boundaries: alphha boundaries, a tensor of shape [total_num_patches, 1, H, W]
    :patches_limits_list: a list with total_num_patches tensors 

    return strokes parameters of shape [budget, n_patches, num_params]
    """
    indices, selected_tensors, n_total = select_tensors_with_n_true(boundaries, patches_limits_list, N)
    strokes_l = []
    
    for i in range(len(patches_limits_list)):
        if i in indices:
            
            strokes = init_boundary_stroke_params(boundaries[i].squeeze(), budget, device)
            strokes_l.append(strokes)
    strokes = torch.stack(strokes_l, dim=1).detach().requires_grad_() # [budget, npatches, 13]
    
    return strokes, indices 


def remove_transparency(stroke, device, num_params=13):
    mask = torch.zeros_like(stroke).to(device)
    if num_params == 13:
        mask[:, 8:10] = 1.0
    else:
        mask[:, 12:14] = 1.0
    stroke = stroke + mask
    return stroke.clip(min=0, max=1)

def make_transparent(stroke, device, idx):
    mask = torch.zeros_like(stroke).to(device)
    mask[idx, 8:10] = 1.0
    stroke = stroke - mask
    return stroke.clip(min=0, max=1)

def clip_width(stroke, num_params, max, device, min=0.01):
    mask = torch.zeros_like(stroke).to(device)
    mask[:, 6:8] = 1.0
    return torch.where(mask > 0, torch.clamp(stroke, min=min, max=max), stroke)

# RENDERER UTILITIES -------------- 

def setup_renderer(args, device):
    rend_path = args.renderer_ckpt_path
    renderer = FCN().to(device)
    # Load renderer 
    renderer.load_state_dict(torch.load(rend_path, map_location=device))
    renderer.eval()
    return renderer


def render(canvas, strokes, budget, brush_size, renderer, num_params=13):
    """Render stroke parameters into canvas
    :param canvas: [n_patches, 3, 128, 128]
    :param strokes: [budget, n_patches, num_params]
    """
    
    # Iterate over budget and render strokes one by one 
    for t in range(budget):

        # Get stroke at timestep t 
        stroke_t = strokes[t]#.unsqueeze(0) # [num_patches, 13]

        # render 
        canvas = forward_renderer(stroke_t, canvas, brush_size, num_params, renderer, device)

    return canvas, None, None


def forward_renderer(stroke, canvas, brush_size, num_params, renderer, device):
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
    stroke = clip_width(stroke, num_params, max=brush_size, device=device)

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


# VISUALIZATION AND OTHER UTILITIES ------ 

def draw_bboxes(img, boxes_loc, level, writer, canvas=True, path=None):
    img = img.clone()
    #print(f'Drawing bboxes, img shape: {img.shape}. Canvas? {canvas}')

    thick = 4

    red = Color('red')
    blue = Color('blue')
    color_range = list(red.range_to(blue, len(boxes_loc)))

    for i in range(len(boxes_loc)):
        curr_patch_loc = boxes_loc[i]  # tuple of (start_h, end_h)(start_w,end_w)
        h_st = curr_patch_loc[0][0]
        h_end = curr_patch_loc[0][1]
        w_st = curr_patch_loc[1][0]
        w_end = curr_patch_loc[1][1]

        R = color_range[i].red
        G = color_range[i].green
        B = color_range[i].blue

        # upper horizontal
        img[:, 0, h_st:h_st + thick, w_st:w_end] = R #1 # R
        img[:, 1, h_st:h_st + thick, w_st:w_end] = G #0 # G
        img[:, 2, h_st:h_st + thick, w_st:w_end] = B #0 # B

        # lower horizontal
        img[:, 0, h_end - thick:h_end, w_st:w_end] = R #1
        img[:, 1, h_end - thick:h_end, w_st:w_end] = G #0
        img[:, 2, h_end - thick:h_end, w_st:w_end] = B #0

        # left vertical
        img[:, 0, h_st:h_end, w_st:w_st + thick] = R #1
        img[:, 1, h_st:h_end, w_st:w_st + thick] = G #0
        img[:, 2, h_st:h_end, w_st:w_st + thick] = B #0

        # right vertical
        img[:, 0, h_st:h_end, w_end - thick:w_end] = R #1
        img[:, 1, h_st:h_end, w_end - thick:w_end] = G #0
        img[:, 2, h_st:h_end, w_end - thick:w_end] = B #0

    if canvas:
        writer.add_image(f'high_error_canvas_level_{level}', img_tensor=img.squeeze(), global_step=0)
    else:
        writer.add_image(f'high_error_ref_img_level_{level}', img_tensor=img.squeeze(), global_step=0)

    if path != None:
        img_name = f'high_error_canvas_level_{level}.png'
        img_name_complete = os.path.join(path, img_name)
        save_image(img.squeeze(), img_name_complete)


def pad_crop(crop, patch_limit, H, W):
    """
    Given a blended crop (the individual canvas), pad tithem with respect to its locations in the general canvas 
    """
    h_st = patch_limit[0][0]
    h_end = patch_limit[0][1]
    w_st = patch_limit[1][0]
    w_end = patch_limit[1][1]

    # Padding lengths 
    pad_left = w_st 
    pad_right = W - w_end
    pad_top = h_st
    pad_bottom = H - h_end

    padding_sizes = (pad_left, pad_right, pad_top, pad_bottom)

    out = F.pad(crop, padding_sizes, 'constant', 0)

    return out 

def blend_padded_canvas(padded_canvas, source_img, general_canvas, first=False, resize_factor=0.5):
    """
    Blend padded canvas onto the general canvas. First, decrease the size of all 3 tensors 
    :padded_canvas: a tensor of shape [1, C, H, W]
    :source_img: a tensor of shape [1, C, H, W]
    :general_canvas: a tensor of shape [1, C, H, W]
    :first: flat to indicate if it's the first time blending. 
        - If True -> decrease its size
        - If False -> Don't change its size 
    All three tensors have the same shape 
    """
    # 1. Decrease all three tensors 
    n_padded_canvas = F.interpolate(padded_canvas, scale_factor=resize_factor)
    if first:
        source_img = F.interpolate(source_img, scale_factor=resize_factor)
        general_canvas = F.interpolate(general_canvas, scale_factor=resize_factor)

    # 2. Blend them 
    alpha = torch.where(n_padded_canvas > 0, 1, 0)  # [1, 3, H, W] White where strokes are, black in the background 
    alpha = alpha[:, 0, :, :].unsqueeze(1) # [1, 1, H, W]
    #blend_canvas = n_padded_canvas + general_canvas
    general_canvas = general_canvas * (1 - alpha) + n_padded_canvas

    return general_canvas, source_img

def blend_all_canvases(canvases, patches_limits, general_canvas, source_img, logger, resize_factor):
    """
    Blends all canvases into a bigger general_canvas, decrase their sizes for a global loss (CLIP)
    :param canvases: a tensor of shape [n_patches, 3, 128, 128]
    :param patches_limits: a list of n_patches tuples of shape ((h_st, h_end), (w_st, w_end))
    :param general_canvas: the global big canvas of shape [1, 3, H, W]
    """

    N = len(patches_limits)
    H = general_canvas.shape[2]
    W = general_canvas.shape[3]

    for i in range(N):
        patch_limit = patches_limits[i]
        padded_crop = pad_crop(canvases[i].unsqueeze(0), patch_limit, H, W)
        first = False
        if i == 0:
            logger.add_image(f'padded_canvas_{i}', img_tensor=padded_crop.squeeze(), global_step=0)
            first = True
       
        general_canvas, n_source_img = blend_padded_canvas(padded_crop, source_img, general_canvas, first=first, resize_factor=resize_factor)
        if first:
            dec_sourc_img = n_source_img

    return general_canvas, dec_sourc_img


def compose_general_canvas(args, canvas, mode, patches_limits, npatches_w, general_canvas, blendin=True):
    """Takes all canvas patches and stiches them together 
    :param canvas: a tensor of shape [npatches, 3, 128, 128]
    """
    
    if args.patch_strategy_detail == 'natural' and mode == 'natural':
        #self.general_canvas = RU.blend_diff(canvas, patches_limits, self.general_canvas, alpha=1.0)
        general_canvas = RU.blend_general_canvas_natural(canvas.detach(), patches_limits, general_canvas=general_canvas, blendin=blendin)

    else:
        #self.general_canvas = RU.blend_diff(canvas, patches_limits, self.general_canvas, alpha=1.0)
        general_canvas = RU.blend_general_canvas(canvas.detach(), general_canvas, args, patches_limits, npatches_w) # self.general_canvas [1, 3, H, W]

    return general_canvas