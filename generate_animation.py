import os
import argparse
from glob import glob 
import numpy as np
from PIL import Image
import imageio
import torch
import torch.nn.functional as F
import gc
import pdb 
import torchvision.transforms as transforms 
from torch.autograd import profiler
from colour import Color
import re

device = torch.cuda.current_device() # 

def extract_name(s):
    pattern = r'([^_/]+)_mask_'
    match = re.search(pattern, s)
    if match:
        return match.group(1)
    else:
        return None

def find_initial_image(directory, level):
    # Search for files with the specified pattern
    name = extract_name(directory)
    filename = os.path.join(directory, f'{name}_lvl_{level}.jpg')
    #matching_files = glob(search_pattern)
    
    # Filter out files with "_texture_" in their name
    #matching_files = [f for f in matching_files if "_texture_" not in f]

    # Return the first matching file, or None if no matches found
    #return matching_files[0] if matching_files else None
    return filename


def print_cuda_memory():
    device = torch.cuda.current_device()
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
    reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)  # Convert to MB

    print(f"Device {device}: Allocated memory = {allocated_memory:.2f} MB, Reserved memory = {reserved_memory:.2f} MB")


def combine_gifs(input_dir, output_file, fps=30):
    images = []
    gif_files = sorted(glob(os.path.join(input_dir, "*.gif")))

    for gif_file in gif_files:
        gif = imageio.get_reader(gif_file)
        for frame in gif:
            images.append(frame)

    imageio.mimsave(output_file, images, fps=fps)

def save_canvas_to_image2(canvas, output_dir, frame_number):
    canvas = canvas.squeeze(0)  # Remove the batch dimension
    canvas_np = canvas.permute(1, 2, 0).detach().cpu().numpy()
    img = Image.fromarray((canvas_np * 255).astype(np.uint8))
    img.save(os.path.join(output_dir, f"frame_{frame_number:04d}.png"))


def save_canvas_to_image(canvas, output_dir, frame_number, src_img=None):
    canvas_np = canvas.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
    half_size = True
    # If src_img is provided, concatenate it to the left of the canvas
    if src_img:
        src_image = Image.open(src_img)

        if half_size:
            # Resize the source image to half the canvas height and compute its width
            target_height = canvas_np.shape[0] // 2
            aspect_ratio = src_image.width / src_image.height
            target_width = int(aspect_ratio * target_height)
            
            src_image = src_image.resize((target_width, target_height))
            src_image_np = np.array(src_image)

            # Create a black area with half the canvas's width and the same height as the canvas
            black_area_np = np.zeros((canvas_np.shape[0], canvas_np.shape[1] // 2, 3))
            
            # Replace the upper part of the black area with the source image
            black_area_np[:target_height, :target_width] = src_image_np
            
            # Concatenate the images
            canvas_np = np.concatenate((black_area_np, canvas_np), axis=1)

        else:
            src_image = src_image.resize((canvas_np.shape[1], canvas_np.shape[0]))  # Resize to match the canvas's height
            src_image_np = np.array(src_image)
            
            # Concatenate the images
            canvas_np = np.concatenate((src_image_np, canvas_np), axis=1)

    
    imageio.imsave(os.path.join(output_dir, f"frame_{frame_number:04d}.png"), (canvas_np).astype(np.uint8))
    del canvas_np
    gc.collect()


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


def make_binary_alpha(alpha, threshold=0.5):
    """
    Converts the given alpha tensor to binary using the provided threshold.

    Parameters:
    - alpha (torch.Tensor): The input alpha tensor.
    - threshold (float): The value above which alpha values will be set to 1, and below which they will be set to 0.

    Returns:
    - torch.Tensor: A binary alpha tensor.
    """
    binary_alpha = alpha.clone()
    binary_alpha[binary_alpha >= threshold] = 1
    binary_alpha[binary_alpha < threshold] = 0
    return binary_alpha

def add_color_border(stroke, alpha, color_range, i):
    # Assuming stroke is a tensor with shape [1, 3, H, W]
    
    stroke_with_border = stroke.clone()
    alpha_with_border = alpha.clone()
    thickness = 6

    R = color_range[i].red
    G = color_range[i].green
    B = color_range[i].blue
   
    # Setting the red channel to 1 (255 in case of uint8 tensor) for the border
    stroke_with_border[:, 0, :thickness, :] = R # Top border
    stroke_with_border[:, 0, -thickness:, :] = R # Bottom border
    stroke_with_border[:, 0, :, :thickness] = R  # Left border
    stroke_with_border[:, 0, :, -thickness:] = R # Right border
    
    # Setting the green and blue channels to 0 for the border
    stroke_with_border[:,1, :thickness, :] = G  # Top border
    stroke_with_border[:,1, -thickness:, :] = G # Bottom border
    stroke_with_border[:,1, :, :thickness] = G  # Left border
    stroke_with_border[:,1, :, -thickness:] = G # Right border

    # Setting the green and blue channels to 0 for the border
    stroke_with_border[:,2, :thickness, :] = B  # Top border
    stroke_with_border[:,2, -thickness:, :] = B# Bottom border
    stroke_with_border[:,2, :, :thickness] = B  # Left border
    stroke_with_border[:,2, :, -thickness:] = B # Right border

    # Setting the alpha
    alpha_with_border[:,:, :thickness, :] = 1  # Top border
    alpha_with_border[:,:, -thickness:, :] = 1 # Bottom border
    alpha_with_border[:,:, :, :thickness] = 1  # Left border
    alpha_with_border[:,:, :, -thickness:] = 1 # Right border

    return stroke_with_border, alpha_with_border


def process_strokes_and_save(stroke_dict, canvas, output_dir, start_frame=0, everyother=1, src_img=None):
    
    frame_number = start_frame
    print_cuda_memory()
    H, W = canvas.shape[2], canvas.shape[3]
    to_pil = transforms.ToPILImage()
    
    red = Color('red')
    blue = Color('blue')
    color_range = list(red.range_to(blue, len(stroke_dict.values())))

    p = 0
    for patch_strokes in stroke_dict.values():
        i = 0
        
        for stroke, location, rgb in patch_strokes:
            
            if i%everyother == 0: # a way to make it less expensive 

                alpha, color_stroke = stroke[0] # stroke is a list of list 

                # img_before = to_pil(alpha.cpu().squeeze())
                # img_before.save(os.path.join(output_dir, f"alpha_{frame_number:04d}.png"))

                alpha = make_binary_alpha(alpha) # delete darker borders  # [1, 1, 128, 128]
                color_stroke = alpha * rgb.view(-1, 3, 1, 1) # [1, 3, 128, 128]

                # Add border for visualization
                bordered_color_stroke, bordered_alpha = add_color_border(color_stroke, alpha, color_range, p)
                
                # img_before = to_pil(alpha.cpu().squeeze())
                # img_before.save(os.path.join(output_dir, f"alpha_{frame_number:04d}.png"))    

                # Pad  
                padded_bordered_color_stroke = pad_crop(bordered_color_stroke, location, H, W)
                padded_bordered_alpha = pad_crop(bordered_alpha, location, H, W)
                
                padded_alpha = pad_crop(alpha, location, H, W)
                padded_color_stroke = pad_crop(color_stroke, location, H, W)
                
                # Move to device 
                padded_alpha = padded_alpha.to(canvas.device)
                padded_color_stroke = padded_color_stroke.to(canvas.device)
                padded_bordered_color_stroke = padded_bordered_color_stroke.to(canvas.device)
                padded_bordered_alpha = padded_bordered_alpha.to(canvas.device)
                
                # Update the canvas using the blending formula
                temp_canvas = canvas.clone() # Temporary canvas for visualization
                temp_canvas = temp_canvas * (1 - padded_alpha) + (padded_color_stroke * padded_alpha)
                temp_canvas_with_border = temp_canvas.clone()
                
                # Add the bordered color stroke on top for visualization
                mask = (padded_bordered_alpha > 0).float()
                temp_canvas_with_border = temp_canvas_with_border * (1 - mask) + (padded_bordered_color_stroke * mask)

                # Save the intermediate canvas as an image
                save_canvas_to_image(temp_canvas_with_border, output_dir, frame_number, src_img)
                
                # Update the canvas using the blending formula
                canvas = canvas * (1 - padded_alpha) + (padded_color_stroke * padded_alpha)
                canvas = torch.clamp(canvas, 0, 1)
                frame_number += 1

                # Clear GPU memory
                del padded_alpha, padded_color_stroke
                torch.cuda.empty_cache()

                print('frame number: ', frame_number)
                print_cuda_memory()
            i += 1
        p += 1
    print(f'Strokes for gif saved in: {output_dir}')
    return frame_number

def create_gif(input_dir, output_file, fps=30):
    images = []
    image_files = sorted(os.listdir(input_dir))

    for filename in image_files:
        images.append(imageio.imread(os.path.join(input_dir, filename)))

    imageio.mimsave(output_file, images, duration=fps)


def main(input_dir, save_dir, abstracted):
    level_names = ["level_1_gif_package.pth", "level_2_gif_package.pth", "level_3_gif_package.pth"]
    level_files = [os.path.join(input_dir, name) for name in level_names]

    print("abstracted: ", abstracted)

    os.makedirs(save_dir, exist_ok=True)

    k = 0
    level_src_img = 1
    for level_file in level_files:
        if not os.path.exists(level_file):
            print(f"File not found: {level_file}")
            continue

        animation_info = torch.load(level_file)

        if abstracted == False:
            
            initial_img_path = find_initial_image(input_dir, level=k)
            initial_image = Image.open(initial_img_path)
        
            # Convert the image to a tensor
            initial_tensor = torch.FloatTensor(np.array(initial_image).transpose(2, 0, 1)).unsqueeze(0) / 255.0
            
            # Check if GPU is available and move tensor to GPU
            if torch.cuda.is_available():
                initial_tensor = initial_tensor.cuda()
            
            # Use the initial_tensor as the starting canvas
            canvas_gif = initial_tensor

            
        for name, v in animation_info.items():
            
            print('name: ', name)
            
            src_img_name = f'high_error_src_img_{name}_level_{level_src_img}.jpg'
            full_src_img_path = os.path.join(input_dir, src_img_name)

            if abstracted:
                canvas_gif = v['canvas']
            
            strokes_and_locations = v['strokes']    
            num_strokes = v['num_strokes']
            
            if num_strokes > 900:
                everyother = 3
            
            elif num_strokes > 600:
                everyother = 2

            else:
                everyother = 1

            print('Number of strokes: ', num_strokes)
            print('Everyother: ', everyother)

            fullname = f'{name}_{k}'
            gif_dir = os.path.join(save_dir, fullname)
            os.makedirs(gif_dir, exist_ok=True)
            
            # resize_factor = 0.5
            # new_h, new_w = int(canvas_gif.shape[2] * resize_factor), int(canvas_gif.shape[3] * resize_factor)
            # canvas_gif = F.interpolate(canvas_gif, size=(new_h, new_w), mode='bilinear', align_corners=False)

            with torch.no_grad():
                process_strokes_and_save(strokes_and_locations, canvas_gif, gif_dir, start_frame=0, everyother=everyother, src_img=full_src_img_path)

            # Create gif 
            output_file = os.path.join(save_dir, f'{fullname}.gif')
            create_gif(gif_dir, output_file, fps=30)
            
            print(f'{name} ....animation finished')

        k += 1
        level_src_img += 1
        
        # Combine gifs
        # combined_gif_file = os.path.join(save_dir, f'combined_{name}.gif')
        # combine_gifs(gif_dir, combined_gif_file)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create and save gifs")
    
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .pth files")
    parser.add_argument("--save_dir", type=str, default="output", help="Directory to save the gifs")
    parser.add_argument("--abstract", type=str2bool, required=True, help="Create abstraction gif or start from base layer")

    args = parser.parse_args()
    main(args.input_dir, args.save_dir, args.abstract)
    print('Creating animation...')
