import os 
import pdb 
import json 
import numpy as np 
import torch 
import cv2 
from PIL import Image
import imageio
from skimage.segmentation import find_boundaries
from models.renderers import FCN, FCN_2outputs
from torch.utils.tensorboard import SummaryWriter
import torch 
import torch.nn.functional as F
import torchvision
from colour import Color
from src.segmentation import segment_image
from utils import render_utils as RU
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import seaborn as sns
import math 
from scipy.integrate import quad
from scipy.interpolate import interp1d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 

def save_img(image, dirpath, img_name):
    img_name_complete = os.path.join(dirpath, img_name)
    save_image(image, img_name_complete)



def remove_elements_by_indexes(lst, idxs):
    # Create a new list that contains elements that are not at the specified indexes
    new_lst = [lst[i] for i in range(len(lst)) if i not in idxs]
    
    return new_lst


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

    # name = args.exp_name
    # fullpath = os.path.join(args.log_dir, name)

    logger = SummaryWriter(args.log_dir)
    print(f'logger created in {args.log_dir}')
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


def overlay_image_and_mask(image, color_mask, alpha=0.5, img_normalized=False):
    
    # Normalize the image and color mask arrays to values between 0 and 1
    if img_normalized == False:
        image = image / 255.0
        color_mask = color_mask / 255.0
    
    # Create a composite image by combining the original image and the color mask using the specified alpha value
    composite_image = image * (1.0 - alpha) + color_mask * alpha
    
    # Clip the composite image to the range [0, 1] to avoid overflow or underflow
    composite_image = np.clip(composite_image, 0.0, 1.0)
    
    # Convert the composite image back to 8-bit integers and return it
    return (composite_image * 255).astype(np.uint8)

def get_edges_and_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return edges, contours

def resize_array(array, target_size):
    height, width = target_size
    resized_array = np.empty((height, width), dtype='<U16')
    # Using the nearest-neighbor interpolation method for resizing
    resized_array = cv2.resize(array, (height, width), interpolation=cv2.INTER_NEAREST)
    return resized_array

def process_img(args, img_path, writer, style_img_path=None, resize_value=128, min_width=1800):
    """
    Receives image path, opens and resizes it and returns tensor 
    """

    # Open image and resize it 
    src_img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:,:,::-1] # from BGR to RGB uint8 [H,W,3]
    print(f'Original image size H={src_img.shape[0]} x W={src_img.shape[1]}')
    
    new_h, new_w = src_img.shape[0], src_img.shape[1]
    style_img = None # to return 
    
    # If style image 
    if style_img_path != None:
        style_img = cv2.imread(style_img_path, cv2.IMREAD_COLOR)[:,:,::-1] # from BGR to RGB uint8 [H,W,3]

    # If passing a resizing value 
    if resize_value != None:
        src_img = cv2.resize(src_img, (resize_value, resize_value)) # [0-255]
        new_h, new_w = resize_value
    
    else: 
        # Decrease size if it's too big 
        if src_img.shape[0] > min_width or src_img.shape[1] > min_width:
            
            if src_img.shape[0] > min_width or src_img.shape[1] > 4000:
                args.aspect_ratio_downsample = 4
            
            new_h = src_img.shape[0] // args.aspect_ratio_downsample
            new_w = src_img.shape[1] // args.aspect_ratio_downsample
        
            if args.upsample and (new_h < min_width or new_w < min_width): # Adjust it to be at least over 1000 pixels so the painting is not too small 
                if src_img.shape[0] >= src_img.shape[1]:
                    src_img = image_resize(src_img, height=min_width)
                else:
                    src_img = image_resize(src_img, width=min_width)

                new_h = src_img.shape[0]
                new_w = src_img.shape[1]

        elif args.upsample and (new_h < min_width or new_w < min_width): # Adjust it to be at least over 1000 pixels so the painting is not too small 
            print('upsampling ...')
            if src_img.shape[0] >= src_img.shape[1]:
                src_img = image_resize(src_img, height=min_width)
            else:
                src_img = image_resize(src_img, width=min_width)

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
    
    if style_img_path != None: # resize style image based on the resized input image 
        target_height, target_width, _ = src_img.shape
        style_img = cv2.resize(style_img, (target_width, target_height))
        assert style_img.shape == src_img.shape, "style image not the same shape as source image"
   
    # Calculate segmentation map 
    segm_ids, boundaries, segm_cat_ids, seg_labels, binary_masks_list = 0, 0, 0, 0, []
    segm_mask_color_with_boundaries = 0
    if args.use_segmentation_mask or args.use_segmentation_contours:
        
        categories_json = '/home/manuelladron/projects/npp/stroke_opt_main/stroke_opt_23/utils/coco_panoptic_cat.json'
        with open(categories_json) as f:
            categories = json.load(f)
        
        # Option 1 -> Calculate segmentation mask based on original image size, then resize 
        segm_ids, segm_cat_ids, seg_labels, binary_masks_list = segment_image(img_path, categories, src_img, is_path=True) # all are [H, W] except for binary_mask_list which is a list

        # Find boundaries 
        boundaries = find_boundaries(segm_ids, mode='thick') # [H, W]
        boundaries = np.expand_dims(boundaries, axis=0) # [1, H, W]
        boundaries = torch.from_numpy(boundaries) # [1, H, W] -> Boolean tensor with boundaries selected
        
        segm_mask_color = map_segmentation_mask_to_colors(segm_cat_ids) # [H, W, 3]

        # Overlay for visualization purposes 
        segm_mask_overlay = overlay_image_and_mask(src_img, segm_mask_color, alpha=0.5)
        segm_mask_overlay = torch.from_numpy(segm_mask_overlay.transpose(2,0,1)) # [C, H, W]

        # segmentation mask 
        # segm_mask_color = normalize_img(segm_mask_color) # This was making the image be all black
        segm_mask_color = torch.from_numpy(segm_mask_color.transpose(2,0,1)) # [C, H, W]
        
        # Depict boundaries
        segm_mask_color_with_boundaries = torch.where(boundaries == True, 0.0, segm_mask_color)

        writer.add_image('segm_original', img_tensor=segm_mask_color_with_boundaries, global_step=0)
        writer.add_image('segm_original_overlay', img_tensor=segm_mask_overlay, global_step=0)

        # Save 
        basename = os.path.basename(args.image_path).split(".")[0]
        
        img_name = f'{basename}_segm_mask_overlay_original.jpg'
        save_img(segm_mask_overlay.squeeze()/255., args.save_dir, img_name)
        
        img_name = f'{basename}_segm_mask_original.jpg'
        save_img(segm_mask_color_with_boundaries.squeeze()/255., args.save_dir, img_name)

        segm_mask_color_with_boundaries = segm_mask_color_with_boundaries.to(device)

    if args.use_edges:
        edges, contours = get_edges_and_contours(src_img)
        edges = normalize_img(edges)
        edges = torch.from_numpy(edges).unsqueeze(0) # [C, H, W]
        writer.add_image('edges', img_tensor=edges, global_step=0)
        pdb.set_trace()

    # Normalize it and conver to torch tensor 
    src_img = normalize_img(src_img)
    img = torch.from_numpy(src_img.transpose(2,0,1)).unsqueeze(0) # [1, C, H, W]

    if style_img_path != None:
        style_img = normalize_img(style_img)
        style_img = torch.from_numpy(style_img.transpose(2,0,1)).unsqueeze(0) # [1, C, H, W] 
    
    print(f'Adjusted input image -> H={img.shape[2]}, W={img.shape[3]}')
    return img, mask, npatches_h, npatches_w, segm_ids, boundaries, segm_cat_ids, seg_labels, binary_masks_list, args.aspect_ratio_downsample, style_img


def increase_boundary_thickness(binary_image, kernel_size=3, stride=1, padding=0):
    # Expand the binary image to 3D, with a single channel and height/width equal to the original size
    binary_image = binary_image.unsqueeze(0).unsqueeze(0)

    # Perform max pooling with the specified kernel size, stride, and padding
    dilated_image = F.max_pool2d(binary_image, kernel_size, stride, padding)

    # Squeeze the result back to 2D
    dilated_image = dilated_image.squeeze().squeeze()

    return dilated_image


# PATCHING UTILITIES -------------- 
def merge_tensors_uniform(tensor_a, tensor_b, indices_a, indices_b):
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
    #assert N == len(indices_a)
    #assert (M - N) == len(indices_b)
    
    for i in range(len(indices_a)): # 0, 1, 2....
        # indices_a[i] = 37, 68, 3, 15, ...
        #print(f'i: {i}, indices_a[i]: {indices_a[i]}')
        merged_tensor[indices_a[i]] = tensor_a[i]

    for j in range(M - len(indices_a)):
        #print(f'j: {j}, indices_b[j]: {indices_b[j]}')
        merged_tensor[indices_b[j]] = tensor_b[indices_b[j]]

    return merged_tensor 


def check_binary_tensor(tensor, N):
    # Count the number of True values in the tensor
    num_true = torch.sum(tensor).item()

    # Check if the number of True values is greater than N
    if num_true > N:
        return True
    else:
        return False

def select_tensors_with_n_true(tensor_list, limits_list, N):
    """
    Selects indexes of tensors that at least have >= N true/1s values 
    :returns: a list with selected indices and total number of selected tensors 
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


def get_patches_w_overlap(args, img, npatchesH, npatchesW, writer=None, name='src_img', is_mask=False):
    """
    Scan an image with a step size and stores its corresponding patches
    :param img: general size image of shape [1, C, H, W]

    :return: list with patches, and corresponding list with the patches boundaries in the format ((h_st, h_end), (w_st, w_end))
    """
    patches = []
    patches_limits = []  # Also for mask 
    mask_in_patch = [] # Boolean list to indicate whether there is a mask or not 
    
    for h in range(npatchesH): 
        start_h = max((128 * h) - (args.overlap*h), 0) # so it doesn't get negative numbers
        
        for w in range(npatchesW):
            start_w = max((128 * w) - (args.overlap*w), 0) # so it doesn't get negative numbers
            end_w = 128 + (w * (128 - args.overlap))
            end_h = 128 + (h * (128 - args.overlap))
            
            patch = img[:, :, start_h:end_h, start_w:end_w]
            patches.append(patch)
            patches_limits.append([(start_h,end_h),(start_w,end_w)])

            if is_mask:
                # Check if this patch has 1s 
                has_true_values = check_binary_tensor(patch, N=100) # If more than N pixels = True, then returns true 
                
                # Add True / False to the list 
                mask_in_patch.append(has_true_values)
                
    if writer != None:
        img_grid = torchvision.utils.make_grid(torch.cat(patches,dim=0), nrow=npatchesW)
        writer.add_image(f'{name}_by_patches', img_tensor=img_grid, global_step=0)
    
    return patches, patches_limits, mask_in_patch

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


def check_patch(mask: torch.Tensor, x: int, y: int, threshold: float) -> bool:
    """
    Creates a 128x128 patch around the center point (x, y) and returns true if
    80% of the pixels of this patch are inside the binary mask.
    
    Args:
    - mask: A binary mask of shape [1, 1, H, W]
    - x: x-coordinate of the center point
    - y: y-coordinate of the center point
    
    Returns:
    - A boolean indicating if 80% of the pixels in the patch are inside the mask.
    """
    H, W = mask.shape[-2:]
    half_size = 64
    patch_top = max(0, y - half_size)
    patch_left = max(0, x - half_size)
    patch_bottom = min(H, y + half_size)
    patch_right = min(W, x + half_size)
    
    patch = mask[..., patch_top:patch_bottom, patch_left:patch_right]
    patch = F.pad(patch, (64 - (x - patch_left), 64 - (y - patch_top),
                          64 - (patch_right - x), 64 - (patch_bottom - y)))
    num_pixels_in_patch = 128 * 128
    num_pixels_in_mask = patch.sum()
    if num_pixels_in_mask >= threshold * num_pixels_in_patch:
        return True
    else:
        return False



def create_N_random_patches(image, N, mask=None, threshold=50, patch_size=128):
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
            #inmask = check_patch(mask, cx, cy, threshold=0.4) # 50% pixels should be in the mask 
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


def visualize_progress(canvases, active_canvases, general_canvas, mask, indices):
    """
    :param canvas: [N, 3, 128, 128], N is all patches 
    :param general_canvas: [1, 3, H, W], overall canvas 
    :param mask: [M, 3, 128, 128], M << N is patches that correspond to segmentation mask 
    :param active_canvases: [M, 3, 128, 128], M << N is patches that correspond to segmentation mask 
    :indices: list of integers corresponding to where patches have a segementation mask  
    """
    return 0 


def high_error_candidates(canvas, patches_list, patches_loc_list, level, num_patches_this_level, name=''):
    """
    Resets the number of total patches based on how many quasi non-overlapping patches we can find

    :param canvas: [1, 3, H, W] general canvas
    :param patches_loc_list: list of length N with coordinate patches tuples (start_h, end_h), (start_w, end_w)
    :param patches_list: list of length N (from get random N patches) with patches of shape [3, 128, 128]
    
    :return: filtered selected patches 
    """

    # 1) crop canvas with patches loc
    # 2) calculate error maps
    # 3) sort and select topk
    
    patches = [] # crops patches from canvas 
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
    if k > len(errors): 
        k = len(errors) #- 1
    total_number_patches = k

    values, indices = torch.topk(errors, k=k)
    selected_patches_loc = [*map(patches_loc_list.__getitem__, indices.tolist())] # list with the selected patches limits (I think it replaces a loop)
    selected_patches_img = torch.index_select(patches_img, 0, indices).to(device) # select patches based on indixes along dimension 0 
    # if name == 'person':
    #     pdb.set_trace()
    return selected_patches_loc, selected_patches_img, indices, values

def get_natural_patches(number_uniform_patches, source_img, general_canvas, general_canvas_txt, logger, level, mask, K_number_natural_patches, path=None, name=''):
    """
    ****** Visual working memory in the paper *******

    Gets source_image patches with a high-error map algorithm and returns those patches, their location, indices of where they are, 
    error value and mask patches if there is any mask 

    Args 
        number_uniform_patches: int 
        source_img:     a tensor of shape [1, 3, H, W]
        general_canvas: a tensor of shape [1, 3, H, W]
        level:          an int indicating the painting level 
        mask:           a tensor of shape [1, 1, H, W]
        K_number_natural_patches: an int 

    Returns: the top N high-est error target patches, their location, selected indices and values, and the mask patches
    """
    print(f'\n Getting natural patches: {name}')
    
    # 1) Start with a slightly smaller number P, than the uniform approach (not using it now)
    num_patches = int(number_uniform_patches / 1.0)
    
    # Generate a list of valid patches T < P which if a mask is provided, they have to fall within the mask. 
    list_patches_locations = create_N_random_patches(source_img, num_patches, mask=mask) # list with lists of tuples 

    # Crop those patches in the source image 
    list_natural_src_patches = crop_image(list_patches_locations, source_img, return_tensor=False)
    
    # 2) Pseudo-Attention: compute high level error between the list of patches and sort them to paint it by priority 
    print(f'Calculating high error candidates')
    
    if list_patches_locations == []:
        print(f'No patches for this {name} mask')
        return -1, -1, -1, -1, -1
    
    print(f'\nName: {name}, num_natural_patches: {K_number_natural_patches}')
    
    # 3) Crops those patches in the general canvas and calculate error between them, return maximum K_number_natural_patches
    patches_limits, target_patches, indices, values = high_error_candidates(general_canvas, list_natural_src_patches, list_patches_locations, 
                                                                            level, K_number_natural_patches, name)
    
    if patches_limits == []:
        print(f'No patches for this {name} mask after calculating high error candidates')
        return -1, -1, -1, -1, -1   
    
    # Note: indices correspond to the selected indices from the pre-selection given by list_patches_locations
    mask_patches = None
    if mask != None:
        mask_patches = crop_image(patches_limits, mask, return_tensor=True).to(device)

    # Log boxes in canvas and source image  
    draw_bboxes(general_canvas_txt, patches_limits, level, logger, name, canvas=True, path=path)
    draw_bboxes(source_img, patches_limits, level, logger, name, canvas=False, path=path)   # [npatches, 3, 128,128]
    
    return target_patches, patches_limits, indices, values, mask_patches 


# STROKES UTILITIES -------------- 


def remove_transparency(stroke, num_params=13):
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

def clip_width(stroke, num_params, max, device=device, min=0.01):
    mask = torch.zeros_like(stroke).to(device)
    mask[:, 6:8] = 1.0
    return torch.where(mask > 0, torch.clamp(stroke, min=min, max=max), stroke)

# RENDERER UTILITIES -------------- 

def setup_renderer(args, device):
    if args.brush_type == 'curved':
        rend_path = args.renderer_ckpt_path
        renderer = FCN().to(device)
    else:
        rend_path = args.renderer_ckpt_path_straight
        renderer = FCN_2outputs(8).to(device)
   
    # Load renderer 
    renderer.load_state_dict(torch.load(rend_path, map_location=device))
    renderer.eval()
    return renderer

def any_column_true(A):
    """
    :param A: 2D tensor of [rows, cols]
    returns a 1D tensor of shape [rows] with False if all columns of A are False
    """
    return torch.any(A, dim=1)

def render_with_filter(canvas, strokes, patch_indices, stroke_indices, brush_size, mask, 
                        renderer, level, mode, writer=None, texturize=False, painter=None, segm_name='', 
                        second_canvas=None, third_canvas=None, use_transp=True, patches_limits=None, general_canvas=None):
    
    """Render stroke parameters into canvas, and optionally into a second and third canvases (for segmentation masks)
    
    :param canvas: A tensor of patches that correspond to a segmentation mask [n_patches, 3, 128, 128]
    :param strokes: [budget, n_patches, num_params]
    :param patch_indices: len with integers of valid patches 
    :param stroke_indices: bool [budget, n_patches]
    :param patches_limits: list with location of stroke[(xmin,ymin), (xmax,ymax)]

    :param mask: [num_patches, 1, 128, 128]
    """
    strokes = strokes.permute(1,0,2) # [npatches, budget, num_params]
    stroke_indices = stroke_indices.permute(1,0) # [npatches, budget]

    npatches = strokes.shape[0]
    budget = strokes.shape[1]
    num_params = strokes.shape[2]

    assert budget == stroke_indices.shape[1]
    assert npatches == stroke_indices.shape[0]
    
    valid_patches = any_column_true(stroke_indices) # [npatches]

    new_canvases = []
    new_second_canvases = []
    new_third_canvases = []

    i_til_rgb = 10 if num_params == 13 else 8

    all_strokes_4_gif = dict()

    num_valid_strokes = 0
    for i in range(npatches):
        all_strokes_4_gif[str(i)] = []

        #print(f'npatch: {i} / {npatches}')
        patch_limits_this_canvas = patches_limits[i]
        
        this_canvas = canvas[i].unsqueeze(0) # [1, 3, 128, 128]
        if second_canvas != None:
            this_second_canvas = second_canvas[i].unsqueeze(0)
        if third_canvas != None:
            this_third_canvas = third_canvas[i].unsqueeze(0)

        # Patch with at least one valid stroke 
        if valid_patches[i] != False:
            
            strokes_patch = strokes[i] # [budget, num_params]
            # canvas here is the same 
            for j in range(budget):

                #print(f'stroke number: {j} / {budget}')
                stroke_t = strokes_patch[j].unsqueeze(0) # [1, num_params]
                rgb = stroke_t[:, 10:]
                
                # This is a valid stroke and should be within the mask 
                if stroke_indices[i, j]:
                    assert torch.all(stroke_t != -1), "Wrong stroke, check the filter algorithm" # Double check it's not a bad stroke 
                    num_valid_strokes += 1
                    
                    if texturize:
                        this_canvas, _ ,_, _  = RU.texturize(stroke_t, this_canvas, brush_size, j, num_params, writer, level, painter=painter, segm_name=segm_name, use_transp=use_transp, patches_limits=None, general_canvas=None)
                        if second_canvas != None:
                            this_second_canvas, _ ,_, _  = RU.texturize(stroke_t, this_second_canvas, brush_size, j, num_params, writer, level, painter=painter, segm_name=segm_name, use_transp=use_transp, patches_limits=None, general_canvas=None)
                        
                        # Third canvas is progress of the painting 
                        if third_canvas != None:
                            try:
                                #pdb.set_trace()
                                this_third_canvas, _ ,_, strokes_gif  = RU.texturize(stroke_t, this_third_canvas, brush_size, j, num_params, writer, level, painter=painter, segm_name=segm_name, use_transp=use_transp, patches_limits=None, general_canvas=general_canvas) # patches_limits=patch_limits_this_canvas,
                                #all_strokes_4_gif[str(i)].append((strokes_gif, patch_limits_this_canvas))
                                #pdb.set_trace()
                            
                            except: 
                                print(f'error in npatch: {i} / {npatches} | stroke number: {j} / {budget}')
                                #pdb.set_trace()
                                continue 
                        # strokes_gif is a list of 1, containing another list of [alpha_stroke, color_stroke]

                    else:
                        this_canvas = forward_renderer(stroke_t, this_canvas, brush_size, num_params, renderer, device, mask=mask[i].unsqueeze(0), use_transp=use_transp, patches_limits=None, general_canvas=None) # [if passing mask, apply filterwise]
                        
                        if second_canvas != None:
                            this_second_canvas  = forward_renderer(stroke_t, this_second_canvas, brush_size, num_params, renderer, device, mask=mask[i].unsqueeze(0), use_transp=use_transp, patches_limits=None, general_canvas=None) # [if passing mask, apply filterwise]
                        
                        if third_canvas != None:
                            #print('in main func: ', patch_limits_this_canvas)
                            #try:
                            this_third_canvas, strokes_gif = forward_renderer(stroke_t, this_third_canvas, brush_size, num_params, renderer, device, mask=mask[i].unsqueeze(0), use_transp=use_transp, patches_limits=patch_limits_this_canvas, general_canvas=None) # [if passing mask, apply filterwise]
                            # strokes_gif is a list of 1, containing another list of [alpha_stroke, color_stroke]
                            all_strokes_4_gif[str(i)].append((strokes_gif, patch_limits_this_canvas, rgb)) # pass rgb for a better reconstruction later on when making the gif 
                            #print('\nadded strokes 4 gif!\n')
                            # except: 
                            #     print(f'error in npatch: {i} / {npatches} | stroke number: {j} / {budget}')
                            #     continue 
                           
            # At uniform mode, we need to add canvases only if they are valid 
            if mode == 'uniform':
                new_canvases.append(this_canvas)
                if second_canvas != None:
                    new_second_canvases.append(this_second_canvas)
                if third_canvas != None:
                    new_third_canvases.append(this_third_canvas)
        
        # At natural, we add canvases regardless whether they are valid 
        if mode == 'natural':
            new_canvases.append(this_canvas)
            if second_canvas != None:
                new_second_canvases.append(this_second_canvas)
            if third_canvas != None:
                new_third_canvases.append(this_third_canvas)
    
    if new_canvases != []:
        new_canvases = torch.cat(new_canvases, dim=0)
        #assert new_canvases.shape[0] == len(patch_indices)
        successful = True

        if new_second_canvases != []:
            new_second_canvases = torch.cat(new_second_canvases, dim=0)
        if new_third_canvases != []:
            new_third_canvases = torch.cat(new_third_canvases, dim=0)
    else:
        new_canvases = canvas
        successful = False

        if second_canvas != None:
            new_second_canvases = second_canvas
        if third_canvas != None:
            new_third_canvases = third_canvas
    
    return new_canvases, successful, new_second_canvases, new_third_canvases, all_strokes_4_gif, num_valid_strokes # dictionary with key = patch_i, value = [strokes, patches_limits]


def render(canvas, strokes, budget, brush_size, renderer, num_params=13, level=None, texturize=False, painter=None, writer=None, segm_name='', use_transp=True):
    """Render stroke parameters into canvas
    :param canvas: [n_patches, 3, 128, 128]
    :param strokes: [budget, n_patches, num_params]
    """
    all_strokes_4_gif = dict()
    
    # Iterate over budget and render strokes one by one 
    for t in range(budget):

        # Get stroke at timestep t 
        stroke_t = strokes[t]#.unsqueeze(0) # [num_patches, 13]

        if texturize:
            canvas, _ ,_, _ = RU.texturize(stroke_t, canvas, brush_size, t, num_params, writer, level, painter=painter, segm_name=segm_name, use_transp=use_transp)
        
        else:
            i_til_rgb = 10 if num_params == 13 else 8
            canvas = forward_renderer(stroke_t, canvas, brush_size, num_params, renderer, device, use_transp=use_transp)


    return canvas, None, None


def forward_renderer(stroke, canvas, brush_size, num_params, renderer, 
                    device, mask=None, return_alpha=False, 
                    use_transp=True, patches_limits=None, 
                    general_canvas=None):
    """
    Renderer network takes in all stroke parameters except for RGB. Renderer outputs an alpha stroke of 128x128 with a white background. 
    Stroke black, background white 
    :param stroke: strokes that are being optimized, a tensor of shape [total patches, 13]
    :param canvas: all canvases patches, a tensor of shape [total patches, 3, 128, 128]
    
    :param mask: Apply pixelwise gate for filtering strokes  
    
    """
    i_til_rgb = 10 if num_params == 13 else 8
    width = canvas.shape[2]
    
    original_canvas = canvas.clone()

    strokes_animation = []

    # Make it opaque and clip width -> Transparency is being controlled by 
    if use_transp == False:
        stroke = remove_transparency(stroke, num_params)
        #print('stroke_transparency: ', stroke[:, 8:10])
    
    stroke = clip_width(stroke, num_params, max=brush_size, device=device)

    #print('stroke_transparency: ', stroke[:, 8:10])
    # Get stroke alpha 
    alpha = (1 - renderer(stroke[:, :i_til_rgb])).unsqueeze(1) # white stroke, black_background [n_patches, 128, 128] -> [n_patches, 1, 128, 128]

    # Multiply alpha by RGB params
    color_stroke = alpha * stroke[: , -3:].view(-1, 3, 1, 1) # [N, 3, 128, 128]

    # Reshape alpha 
    alpha = alpha.view(-1, 1, width, width) # [N, 1, 128, 128]
    
    # Blend 
    # 1 - alpha = 0 in the stroke regions, and 1 in the background. It zeroes out the canvas at the stroke region and adds canvas with color stroke, whose bckg = 0
    canvas = canvas * (1-alpha) + color_stroke 
    
    if patches_limits != None:
        """
        H, W = general_canvas.shape[2], general_canvas.shape[3]
        padded_alpha = pad_crop(alpha, patches_limits, H, W)
        #padded_color_stroke = pad_crop(color_stroke, patches_limits, H, W)
        padded_color_stroke = padded_alpha * stroke[: , -3:].view(-1, 3, 1, 1)
        strokes_animation = [[padded_alpha, padded_color_stroke]]
        """
        strokes_animation = [[alpha, color_stroke]]
        return canvas.clip(min=0, max=1), strokes_animation

    # return original canvas if the stroke is outside the mask 
    # TODO: check that we are passing the right mask 
    if mask != None:
        result = alpha * mask # [N, 1, 128, 128]
        if torch.allclose(result, alpha) == False:
            return original_canvas 
    
    if return_alpha:
        return canvas.clip(min=0, max=1), alpha

    return canvas.clip(min=0, max=1)

# VISUALIZATION AND OTHER UTILITIES ------ 

def draw_bboxes(img, boxes_loc, level, writer, name, canvas=True, path=None):
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
        writer.add_image(f'high_error_canvas_{name}_level_{level}', img_tensor=img.squeeze(), global_step=0)
    else:
        writer.add_image(f'high_error_ref_img_{name}_level_{level}', img_tensor=img.squeeze(), global_step=0)

    if path != None:
        if canvas:
            img_name = f'high_error_canvas_{name}_level_{level}.jpg'
        else:
            img_name= f'high_error_src_img_{name}_level_{level}.jpg'
        
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
    :general_canvas: a tensor of shape [1, C, H, W], this function updates this tensor 
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
    # Get alpha to blend 
    alpha = torch.where(n_padded_canvas > 0, 1, 0)  # [1, 3, H, W] White where strokes are, black in the background 
    alpha = alpha[:, 0, :, :].unsqueeze(1) # [1, 1, H, W]
    general_canvas = general_canvas * (1 - alpha) + n_padded_canvas

    return general_canvas, source_img

def blend_all_canvases(canvases, patches_limits, general_canvas, source_img, logger, resize_factor):
    """
    Blends all canvases by padding them and applying a normal blending formula into a bigger general_canvas, decrease their sizes for a global loss (CLIP)
    :param canvases: a tensor of shape [n_patches, 3, 128, 128]
    :param patches_limits: a list of n_patches tuples of shape ((h_st, h_end), (w_st, w_end))
    :param general_canvas: the global big canvas of shape [1, 3, H, W]

    :returns: decreased general_canvas and source image to compute loss 
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

        # Call the blending function. At the first patch, decrease the size of padded canvas, source image and general canvas.
        general_canvas, n_source_img = blend_padded_canvas(padded_crop, source_img, general_canvas, first=first, resize_factor=resize_factor)
        
        if first:
            dec_sourc_img = n_source_img

    return general_canvas, dec_sourc_img


def compose_general_canvas(args, canvas, mode, patches_limits, npatches_w, general_canvas, blendin=True):
    """Takes all canvas patches and stitchs them together 
    :param canvas: a tensor of shape [npatches, 3, 128, 128]
    """
    
    if args.patch_strategy_detail == 'natural' and mode == 'natural':
        #self.general_canvas = RU.blend_diff(canvas, patches_limits, self.general_canvas, alpha=1.0)
        general_canvas = RU.blend_general_canvas_natural(canvas.detach(), patches_limits, general_canvas=general_canvas, blendin=blendin)
    
    else:
        #self.general_canvas = RU.blend_diff(canvas, patches_limits, self.general_canvas, alpha=1.0)
        general_canvas = RU.blend_general_canvas(canvas.detach(), general_canvas, args, patches_limits, npatches_w) # self.general_canvas [1, 3, H, W]

    return general_canvas



def order_list_by_index(main_list, index_list):
    """
    Orders a list of strings based on the index of another list of strings.

    Args:
    main_list (list): The list to be ordered.
    index_list (list): The list containing the order index as strings.

    Returns:
    list: The ordered list.
    """
    index_dict = {index_list[i]: i for i in range(len(index_list))}
    ordered_list = sorted(main_list, key=lambda x: index_dict.get(x, len(main_list)))
    return ordered_list


def order_two_lists_by_index(list1, list2, index_list):
    """
    Orders two lists based on the index of another list.

    Args:
    list1 (list): The first list to be ordered.
    list2 (list): The second list to be ordered based on the order of the first list.
    index_list (list): The list containing the order index.

    Returns:
    tuple: A tuple containing the ordered lists.
    """

    index_dict = {index_list[i]: i for i in range(len(index_list))}
    ordered_list1 = sorted(list1, key=lambda x: index_dict.get(x, len(list1)))
    
    
    pdb.set_trace()
    ordered_list2 = [list2[index_dict.get(x, len(list1))] for x in ordered_list1]
    
    return ordered_list1, ordered_list2

def order_tuple_list_by_index(tuple_list, index_list):
    """
    Orders a list of tuples based on the index of another list.

    Args:
    tuple_list (list): The list of tuples to be ordered based on the first element of each tuple.
    index_list (list): The list containing the order index.

    Returns:
    list: The ordered list of tuples.
    """
    # if not set(index_list).issubset(set([t[0] for t in tuple_list])):
    #     raise ValueError("Index list contains invalid elements")

    index_dict = {index_list[i]: i for i in range(len(index_list))}
    ordered_tuple_list = sorted(tuple_list, key=lambda x: index_dict.get(x[0], len(tuple_list)))
    return ordered_tuple_list


def save_canvas_to_image(canvas, output_dir, frame_number):
    canvas = canvas.squeeze(0)  # Remove the batch dimension
    canvas_np = canvas.permute(1, 2, 0).detach().cpu().numpy()
    img = Image.fromarray((canvas_np * 255).astype(np.uint8))
    img.save(os.path.join(output_dir, f"frame_{frame_number:04d}.png"))


def process_strokes_and_save(stroke_dict, canvas, output_dir, start_frame=0):
    frame_number = start_frame

    for patch_strokes in stroke_dict.values():
        
        for stroke, location in patch_strokes:
            alpha, color_stroke = stroke[0] # stroke is a list of list 
            
            # Pad here 
            H, W = canvas.shape[2], canvas.shape[3]
            padded_alpha = pad_crop(alpha, location, H, W)
            #padded_color_stroke = pad_crop(color_stroke, patches_limits, H, W)
            padded_color_stroke = padded_alpha * color_stroke[: , -3:].view(-1, 3, 1, 1)

            padded_alpha = padded_alpha.to(canvas.device)
            padded_color_stroke = padded_color_stroke.to(canvas.device)

            # Update the canvas using the blending formula
            canvas = canvas * (1 - padded_alpha) + (padded_color_stroke * padded_alpha)

            # Save the intermediate canvas as an image
            save_canvas_to_image(canvas, output_dir, frame_number)
            frame_number += 1

    print(f'Strokes for gif saved in: {output_dir}')
    return frame_number


def create_gif(input_dir, output_file, fps=30):
    images = []
    image_files = sorted(os.listdir(input_dir))

    for filename in image_files:
        images.append(imageio.imread(os.path.join(input_dir, filename)))

    imageio.mimsave(output_file, images, fps=fps)


def cap(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def get_absolute_stroke_coordinates(strokes, patches_limits, general_canvas, ongoing_number_strokes):
    """
    Gets strokes parameters and returns a list of strokes with absolute and normalized coordinates in relation to the overall canvas

    Args:
    strokes (tensor): Strokes of shape [budget, num_patches, 13]
    patches_limits (list): The list patches limits length same as second dimension of strokes
    general_canvas (tensor): General canvas of shape [1, 3, H, W]

    Returns:
    normalized absolute middle x,y coordinates of the strokes in relation to the general canvas
    """

    H, W = general_canvas.shape[2], general_canvas.shape[3]
    
    middle_coords_strokes = []
    areas = []

    total_strokes = 0
    for i in range(len(patches_limits)): # patches_limits are in absolute pixels 
        
        h_st = patches_limits[i][0][0]
        h_end = patches_limits[i][0][1]
        w_st = patches_limits[i][1][0]
        w_end = patches_limits[i][1][1]

        for j in range(strokes.shape[0]): # each stroke individually
            this_stroke = strokes[j, i] # 13-tuple
            
            x0, y0 ,x, y, x2, y2, r0, r2, t0, t2, r, g, b = this_stroke # x == row, y == col (x is the index in the vertical axis, y is the index in the horizontal axis)

            # Get absolute coordinates in relation to the patch (0 min, 127 max)
            x_abs_patch = int(math.floor(x.item() * 127)) 
            y_abs_patch = int(math.floor(y.item() * 127)) 

            x_abs = cap(h_st + x_abs_patch, 0, H)
            y_abs = cap(w_st + y_abs_patch, 0, W)

            assert x_abs <= H, f"x_abs: {x_abs}, H: {H}"
            assert y_abs <= W, f"y_abs: {y_abs}, H: {W}"

            x_norm = x_abs / H
            y_norm = y_abs / W

            area = bezier_area(x0.item(), y0.item(), x.item(), y.item(), x2.item(), y2.item(), r0.item(), r2.item())
            areas.append(area)
            
            #print(f'area: {area}')

            middle_coords_strokes.append([(x_abs, y_abs), (x_norm, y_norm)])
            total_strokes += 1

    ongoing_number_strokes += total_strokes
    return middle_coords_strokes, areas, ongoing_number_strokes


def compute_stroke_distribution(stroke_params, canvas_shape, grid_size):
    """
    Compute the distribution of strokes on a discretized canvas.
    
    Parameters:
    - stroke_params: a list of length num_strokes, where each element is [(x_abs, y_abs), (x_norm, y_norm)]
                     x_abs and y_abs are absolute pixel coordinates,
                     while x_norm, y_norm are normalized coordinates in the range 0-1.
    - canvas_shape: a tuple (C, H, W) indicating the shape of the canvas.
    - grid_size: the size of each grid cell.
    
    Returns:
    - A tensor of shape (H // grid_size, W // grid_size) representing the distribution of strokes on the canvas.
    """
    
    _, H, W = canvas_shape
    grid_H, grid_W = (H // grid_size)+1, (W // grid_size)+1
    grid = torch.zeros((grid_H, grid_W))
    
    # print(f"General canvas -> H: {H} | W: {W}")
    # print(f"GRID SHAPE -> {grid.shape}, cell size: {grid_size}")
    # print(f"grid_H  {grid_H} | grid_W: {grid_W}")
    
    # Use the absolute coordinates from stroke_params to map each stroke to a grid cell
    for stroke in stroke_params:
        x_abs, y_abs = stroke[0] # x is vertical, and y horizontal 
        grid_y, grid_x = int(x_abs // grid_size), int(y_abs // grid_size)
        
        # print(f"Stroke -> x_abs: {x_abs} | y_abs: {y_abs}")
        # print(f"grid_y: {grid_y} | grid_x: {grid_x}")

        grid[grid_y, grid_x] += 1
    
    grid /= grid.sum()
    return grid

def plot_stroke_distribution(stroke_params, areas, canvas_shape, grid_size, save_dir, name, level, number_strokes):
    distribution = compute_stroke_distribution(stroke_params, canvas_shape, grid_size)
    
    x_abs_values = [stroke[0][0] for stroke in stroke_params] # vertical 
    y_abs_values = [stroke[0][1] for stroke in stroke_params] # horizontal 
    
    strokes_seq = list(range(len(areas)))

    # Creating the histogram (distribution)
    plt.figure(figsize=(6,6))
    plt.hist2d(y_abs_values, x_abs_values, bins=(canvas_shape[-2] // grid_size, canvas_shape[-1] // grid_size))
    #plt.gca().invert_yaxis()
    plt.colorbar(label="Number of Strokes")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title(f"Distribution of Strokes on {grid_size}x{grid_size} Grid")
    plt.xlim(0, canvas_shape[2])
    plt.ylim(canvas_shape[1], 0)
    plt.savefig(os.path.join(save_dir, f"stroke_distribution_{name}_lvl_{level}.jpg"))
    plt.close()
    
    # Creating the heatmap
    plt.figure(figsize=(6,6))
    plt.imshow(distribution, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Stroke Density")
    plt.xlabel("X-coordinate (Grid)")
    plt.ylabel("Y-coordinate (Grid)")
    plt.title(f"Heatmap of Strokes on {grid_size}x{grid_size} Grid")
    plt.savefig(os.path.join(save_dir,f"stroke_heatmap_{name}_{level}.jpg"))
    plt.close()

    # KDE 2D plot
    fig, ax = plt.subplots(figsize=(6,6))
    cax = sns.kdeplot(x=y_abs_values, y=x_abs_values, cmap='crest', fill=True, levels=20) # Blues, RdGy
    ax.set_xlim(0, canvas_shape[2])
    ax.set_ylim(canvas_shape[1], 0)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D Kernel Density Estimate of Stroke Distribution')
    cbar = fig.colorbar(cax.collections[0], ax=ax, orientation="vertical")
    cbar.set_label('Density')
    plt.savefig(os.path.join(save_dir, f'kde_distribution_{name}_{level}.jpg'))

    # Stroke size by sequence 
    canvas_size=128*128
    normalized_areas_canvas = [(area / canvas_size) * 100 for area in areas]  # multiplied by 100 for percentage

    plt.figure(figsize=(6, 6))
    plt.plot(strokes_seq, normalized_areas_canvas, '-o' , alpha=0.6) # '-o' creates a line graph with circle markers
    plt.title('Stroke Size by Sequence')
    plt.xlabel('Sequence')
    plt.ylabel('Stroke Size (Area)')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'stroke_size_seq_{name}_{level}.jpg'))

    # Stroke size by location 
    # Normalize area values for coloring
    norm_areas = [(area - min(areas)) / (max(areas) - min(areas)) for area in areas]

    plt.figure(figsize=(6, 6))
    plt.scatter(y_abs_values, x_abs_values, c=norm_areas, cmap='viridis', alpha=0.6)
    plt.gca().invert_yaxis()
    plt.colorbar(label='Normalized Stroke Size (Area)')
    plt.title('Stroke Size by Location')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.xlim(0, canvas_shape[2])
    plt.ylim(canvas_shape[1], 0)
    plt.savefig(os.path.join(save_dir, f'stroke_size_loc_{name}_{level}.jpg'))


    # Define bins for the histogram, note that these are the edges so >0.8 is implied by bin edge 0.8 to 1.0
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    relative_areas = [(area / canvas_size) for area in areas] 
    total_strokes = len(relative_areas)

    # Calculate the histogram data
    counts, _ = np.histogram(relative_areas, bins=bins)
    percentages = (counts / sum(counts)) * 100

    plt.figure(figsize=(6, 6))
    # Create the histogram
    #plt.hist(relative_areas, bins=bins, color='mediumpurple', edgecolor='black')

    # Create labels for each bin
    bin_labels = ['<20%', '20-40%', '40-60%', '60-80%', '>80%']
    bin_centers = [(a + b) / 2 for a, b in zip(bins[:-1], bins[1:])]

    plt.bar(bin_centers, percentages, color='mediumpurple', edgecolor='black', width=0.1)
    plt.xticks(bin_centers, bin_labels)

    # Adding labels and title
    plt.title('Distribution of Stroke Sizes Relative to Canvas Area')
    plt.xlabel('Percentage of Canvas Area Covered')
    plt.ylabel('Number of Strokes')
    plt.savefig(os.path.join(save_dir,f"stroke_rel_area_{name}_{level}.jpg"))
    plt.close()






def bezier_point(t, p0, p1, p2):
    """Calculate a point in a quadratic Bezier curve."""
    return (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2

def radius_at_t(t, r0, r2):
    """Linearly interpolate the radius along the curve."""
    return (1 - t) * r0 + t * r2

def bezier_area(x0, y0, x, y, x2, y2, r0, r2, patch_size=128, num_segments=1000):
    """Approximate the area under a Bezier curve with varying radius."""

    # Scale the normalized parameters to the patch size
    x0, y0, x, y, x2, y2 = [coord * patch_size for coord in [x0, y0, x, y, x2, y2]]
    r0, r2 = r0 * patch_size, r2 * patch_size

    # Define the Bezier points
    # p0 = np.array([x0, y0])
    # p1 = np.array([x, y])
    # p2 = np.array([x2, y2])

    # Discretize the curve
    t_values = np.linspace(0, 1, num_segments)
    #bezier_points = np.array([bezier_point(t, p0, p1, p2) for t in t_values])
    radii = np.array([radius_at_t(t, r0, r2) for t in t_values])

    # Calculate the area under the curve by summing the areas of circles
    # with radii interpolated along the curve
    # (this is an approximation)
    circle_areas = np.pi * radii**2
    total_area = np.trapz(circle_areas, t_values)  # more accurate than summing 
    #total_area = np.sum(areas) * (1 / num_segments)

    return total_area