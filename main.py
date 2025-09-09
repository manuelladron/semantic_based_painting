import time 
import os 
import argparse 
import pdb 
import warnings
warnings.filterwarnings("ignore")
import src.style_config as style_config
from src.painter import Painter 


def create_parser():
    parser = argparse.ArgumentParser(description='Stroke Optimization')

    parser.add_argument('--exp_name', type=str, default = 'exp_320')  
    parser.add_argument('--style', type=str, default = 'expressionist', choices=['realistic', 'painterly', 'abstract', 'expressionist']) 

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
    parser.add_argument('--upsample', type=bool, default = True, help='Enable upsampling of small images')
    parser.add_argument('--min_image_size', type=int, default = 1800, help='Minimum image size for upsampling (set lower to avoid upsampling after resize)')
    parser.add_argument('--aspect_ratio_downsample', type=float, default=3)
    parser.add_argument('--image_resize_factor', type=float, default=1.0, help='Factor to resize input image (e.g., 0.33 for 3x smaller, 0.5 for 2x smaller)')
    parser.add_argument('--image_path', type=str, default = 'images/paris2.jpeg') 
    
    parser.add_argument('--save_dir', type=str, default = './results')
    parser.add_argument('--canvas_size', type=int, default=128)
    parser.add_argument('--canvas_color', type=str, default='black', choices=['black', 'white'])

    # Renderer settings 
    parser.add_argument('--renderer_ckpt_path', type=str, default = './model_checkpoints/renderer.pkl')
    parser.add_argument('--renderer_ckpt_path_straight', type=str, default = '')

    # optimization settings 
    parser.add_argument('--lr', type=float, default = 0.004)
    parser.add_argument('--use_mixed_precision', type=bool, default = True)

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
    parser.add_argument('--style_img_path', type=str, default = None)

    return parser 


def define_styles(style, args):
    """
    Defines style parameters based on desired style input
    :style: string, either painterly or realistic 
    :args: parsed args 
    """
    args.brush_sizes = [0.8, 0.4, 0.2, 0.05] # Same for every style | [0.8, 0.4, 0.2, 0.1] in the paper 
    args.iter_steps = [300, 300, 300, 300] 
    
    # Get style-specific settings
    style_params = style_config.style_parameters.get(style, {})

    for key, value in style_params.items():
        setattr(args, key, value)
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
