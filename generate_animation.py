import os
import argparse
from glob import glob 
import numpy as np
from PIL import Image
import imageio
import torch
import torch.nn.functional as F
import pdb 

device = torch.cuda.current_device()

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

def save_canvas_to_image(canvas, output_dir, frame_number):
    canvas = canvas.squeeze(0)  # Remove the batch dimension
    canvas_np = canvas.permute(1, 2, 0).detach().cpu().numpy()
    img = Image.fromarray((canvas_np * 255).astype(np.uint8))
    img.save(os.path.join(output_dir, f"frame_{frame_number:04d}.png"))


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

def process_strokes_and_save(stroke_dict, canvas, output_dir, start_frame=0, everyother=1):
    frame_number = start_frame
    print_cuda_memory()
    H, W = canvas.shape[2], canvas.shape[3]
    
    # if len(stroke_dict.values()) > 48:
    #     everyother = 3

    # elif len(stroke_dict.values()) > 48:
    #     everyother = 2
    
    # else:
    #     everyother = 1
    
    for patch_strokes in stroke_dict.values():
        i = 0
        for stroke, location in patch_strokes:
            if i%everyother == 0:
                alpha, color_stroke = stroke[0] # stroke is a list of list 

                # Pad here 
                padded_alpha = pad_crop(alpha, location, H, W)
                padded_color_stroke = pad_crop(color_stroke, location, H, W)
                #padded_color_stroke = padded_alpha * color_stroke[: , -3:].view(-1, 3, 1, 1)

                padded_alpha = padded_alpha.to(canvas.device)
                padded_color_stroke = padded_color_stroke.to(canvas.device)

                # Update the canvas using the blending formula
                canvas = canvas * (1 - padded_alpha) + (padded_color_stroke * padded_alpha)

                # Save the intermediate canvas as an image
                save_canvas_to_image(canvas, output_dir, frame_number)
                frame_number += 1

                print('frame number: ', frame_number)
                print_cuda_memory()
            i += 1

    print(f'Strokes for gif saved in: {output_dir}')
    return frame_number

def create_gif(input_dir, output_file, fps=30):
    images = []
    image_files = sorted(os.listdir(input_dir))

    for filename in image_files:
        images.append(imageio.imread(os.path.join(input_dir, filename)))

    imageio.mimsave(output_file, images, duration=fps)


def main(input_dir, save_dir):
    level_names = ["level_1_gif_package.pth", "level_2_gif_package.pth", "level_3_gif_package.pth"]
    level_files = [os.path.join(input_dir, name) for name in level_names]

    k = 0
    for level_file in level_files:
        if not os.path.exists(level_file):
            print(f"File not found: {level_file}")
            continue

        animation_info = torch.load(level_file)

        for name, v in animation_info.items():
            
            print('name: ', name)
            
            canvas_gif = v['canvas']
            strokes_and_locations = v['strokes']    
            num_strokes = v['num_strokes']
            
            if num_strokes > 900:
                everyother = 3
            
            elif num_strokes > 600:
                everyother = 2

            else:
                everyother = 1

            fullname = f'{name}_{k}'
            gif_dir = os.path.join(save_dir, fullname)
            os.makedirs(gif_dir, exist_ok=True)
            
            
            process_strokes_and_save(strokes_and_locations, canvas_gif, gif_dir, start_frame=0, everyother=everyother)

            # Create gif 
            output_file = os.path.join(save_dir, f'{fullname}.gif')
            create_gif(gif_dir, output_file, fps=30)
            
            print(f'{name} ....animation finished')

        k += 1
        # Combine gifs
        # combined_gif_file = os.path.join(save_dir, f'combined_{name}.gif')
        # combine_gifs(gif_dir, combined_gif_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create and save gifs")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .pth files")
    parser.add_argument("--save_dir", type=str, default="output", help="Directory to save the gifs")

    args = parser.parse_args()
    main(args.input_dir, args.save_dir)
    print('Creating animation...')
