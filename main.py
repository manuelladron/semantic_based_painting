import time 
import os 
import argparse 
import pdb 
from src.painter import Painter 


def create_parser():
    parser = argparse.ArgumentParser(description='Stroke Optimization')

    parser.add_argument('--exp_name', type=str, default = 'exp_300')  # exp 134, 135 is GOOD! 
    parser.add_argument('--style', type=str, default = 'painterly', choices=['realistic', 'painterly', 'abstract', 'expressionist']) # exp 134, 135 is GOOD! 

    # strategy settings
    parser.add_argument('--brush_type', type=str, default='curved', choices=['straight', 'curved'])
    
    parser.add_argument('--paint_by_patches', type=bool, default = True)
    parser.add_argument('--global_loss', type=bool, default = False)
    parser.add_argument('--texturize', type=bool, default = True)
    parser.add_argument('--use_transparency', type=bool, default = False)

    parser.add_argument('--compute_stroke_distribution', type=bool, default = False)
    parser.add_argument('--save_animation', type=bool, default = True)
    
    # This is old and probably won't be used again 
    parser.add_argument('--use_segmentation_contours', type=bool, default = False)
    parser.add_argument('--use_edges', type=bool, default = False)

    # Don't touch these parameters 
    parser.add_argument('--start_using_masks', type=int, default = 1)
    parser.add_argument('--start_natural_level', type=int, default = 1)
    parser.add_argument('--overlap', type=int, default=20) 

    # misc settings 
    parser.add_argument('--upsample', type=bool, default = False)
    parser.add_argument('--aspect_ratio_downsample', type=float, default=3)
    parser.add_argument('--image_path', type=str, default = './images/building.png') 
    
    parser.add_argument('--save_dir', type=str, default = './results')
    parser.add_argument('--canvas_size', type=int, default=128)
    parser.add_argument('--canvas_color', type=str, default = 'white', choices=['back', 'white'])

    # Renderer settings 
    parser.add_argument('--renderer_ckpt_path', type=str, default = './model_checkpoints/renderer.pkl')
    parser.add_argument('--renderer_ckpt_path_straight', type=str, default = '')

    # optimization settings 
    parser.add_argument('--lr', type=float, default = 0.004)

    # Stroke settings 
    parser.add_argument('--stroke_init_mode', type=str, default = 'grid', choices=['random', 'grid'])

    # loss settings 
    parser.add_argument('--w_perc', type=float, default = 0)

    # Clip Text-Style Optimization loss settings 
    parser.add_argument('--add_style', type=bool, default=False)
    parser.add_argument('--opt_steps_style', type=int, default=500) # exp_17 -> 350
    parser.add_argument('--style_prompt', type=str, default = 'Starry Night by Vincent Van Gogh')
    parser.add_argument('--style_lambda', type=float, default=900)
    parser.add_argument('--content_lambda', type=float, default=10)
    parser.add_argument('--style_patch_lambda', type=float, default=1000)

    # Style Transfer (Image-style optimization)
    parser.add_argument('--style_transfer', type=bool, default=False)
    parser.add_argument('--st_content_w', type=float, default=0.005) # 0.01 works relatively well, 0.001 works well but produces patches 
    parser.add_argument('--style_img_path', type=str, default = '')

    return parser 


def define_styles(style, args):
    """
    Defines style parameters based on desired style input
    :style: string, either painterly or realistic 
    :args: parsed args 
    """
    args.brush_sizes = [0.8, 0.4, 0.2, 0.05] # Same for every style | [0.8, 0.4, 0.2, 0.1] in the paper 
    args.iter_steps = [300, 300, 300, 300] 
    
    # Realism 
    if style == 'realistic':
        #args.brush_sizes = [0.8, 0.7, 0.7, 0.6]
        args.budgets= [9, 49, 64, 81] # [9, 49, 64, 81]  # abstracted [9, 9, 49]   # painterly [9, 16, 16, 9]  # realistic  [9, 9, 49, 64] # hyperrealistic [9,49,64,81]
        args.number_natural_patches = [40, 60, 60]
        args.patch_strategy_detail = 'uniform'
        args.use_segmentation_mask = False
        args.filter_strokes = False
        args.return_segmented_areas = False
   
    # Abstract 
    elif style == 'abstract':
        args.budgets=[9, 16, 16, 9]
        args.number_natural_patches = [25, 25, 50] # [25, 30, 60]
        args.patch_strategy_detail = 'natural'
        args.use_segmentation_mask = True
        args.filter_strokes = True
        args.return_segmented_areas = True
        args.start_using_masks = 1
        args.start_natural_level = 0
        
    # Expressionist
    elif style == 'expressionist':
        args.iter_steps = [500, 500] 
        args.brush_sizes = [0.8, 0.8]
        args.budgets=[9, 9]
        args.number_natural_patches = [9] # [25, 30, 60]
        args.patch_strategy_detail = 'natural'
        args.use_segmentation_mask = False
        args.filter_strokes = False
        args.return_segmented_areas = False
        args.start_using_masks = -1
        args.start_natural_level = 1

    # Painterly 
    else:
        args.budgets=[9, 16, 16, 9] # [9, 16, 16, 9]
        args.number_natural_patches = [25, 25, 25] # [25, 30, 60] | [25, 25, 50]
        args.patch_strategy_detail = 'natural'
        args.use_segmentation_mask = True
        args.filter_strokes = True
        args.return_segmented_areas = True

    return args


if __name__ == '__main__':
    args = create_parser().parse_args()
    
    # Define style 
    args = define_styles(args.style, args)

    print(f'\n------Style: {args.style}--------\n')

    basename = os.path.basename(args.image_path).split(".")[0]
    usemask = str(args.use_segmentation_mask)
    mode = str(args.patch_strategy_detail)
    
    args.exp_name = args.exp_name + f'_{basename}_mask_{usemask}_mode_{mode}_filter_{args.filter_strokes}_brush_sizes_{args.brush_sizes}_budgets_{args.budgets}_st_{args.style_transfer}_content_w_{args.st_content_w}'
    
    # Make dir to save images 
    save_dir = os.path.join(args.save_dir, 'imgs')
    log_dir = os.path.join(args.save_dir, 'logs')
    
    args.save_dir = os.path.join(save_dir, args.exp_name)#, basename)
    args.log_dir = os.path.join(log_dir, args.exp_name)#, basename)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Initialize painter 
    P = Painter(args)
    # Call main function and measure time  # 
    st = time.time()
    P.paint()
    end = time.time()
    print(f'total time: {(end-st)/60} minutes')
