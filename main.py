import time 
import os 
import argparse 
import pdb 
from src.painter import Painter 


def create_parser():
    parser = argparse.ArgumentParser(description='Stroke Optimization')

    parser.add_argument('--exp_name', type=str, default = 'exp_183_marble') # exp 134, 135 is GOOD! 
    
    # strategy settings
    parser.add_argument('--brush_type', type=str, default='curved', choices=['straight', 'curved'])
    
    parser.add_argument('--paint_by_patches', type=bool, default = True)
    parser.add_argument('--global_loss', type=bool, default = False)
    parser.add_argument('--texturize', type=bool, default = True)
    parser.add_argument('--use_transparency', type=bool, default = False)
    
    parser.add_argument('--use_segmentation_contours', type=bool, default = False)

    # Segmentation 
    parser.add_argument('--use_segmentation_mask', type=bool, default = True) # True
    parser.add_argument('--filter_strokes', type=bool, default = True) # true
    parser.add_argument('--return_segmented_areas', type=bool, default = True) # true 

    parser.add_argument('--use_edges', type=bool, default = False)

    parser.add_argument('--start_using_masks', type=int, default = 1)
    parser.add_argument('--start_natural_level', type=int, default = 1)
    parser.add_argument('--patch_strategy_detail', type=str, default='natural', choices=['uniform', 'natural'])

    parser.add_argument('--overlap', type=int, default=20) 

    parser.add_argument('--brush_sizes', type=list, default=[0.7, 0.5, 0.4, 0.05]) # abstracted [0.8, 0.5, 0.2] # realistic [0.8, 0.5, 0.2, 0.05]
    parser.add_argument('--budgets', type=list, default=[9, 16, 49, 64])  # abstracted [9, 9, 49]       [9, 9, 9, 16]          # realistic  [9, 9, 49, 64] # [16, 9, 16, 25]
    parser.add_argument('--iter_steps', type=list, default=[300, 300, 300, 300]) # [300, 300, 300]

    parser.add_argument('--number_natural_patches', type=int, default=[40, 30, 25])  # [25, 30, 25] # [30, 50, 60] WE DID NOT DEFINE THIS IN THE PAPER 

    # misc settings 
    parser.add_argument('--upsample', type=bool, default = True)
    parser.add_argument('--aspect_ratio_downsample', type=float, default=3)
    parser.add_argument('--image_path', type=str, default = '/home/manuelladron/projects/npp/stroke_opt_main/images/buildings/marble_detail.jpeg') # /home/manuelladron/projects/npp/sigg-asia-imgs/siggraph_asia/landscapes/dalle_1.png
    
    parser.add_argument('--save_dir', type=str, default = '/home/manuelladron/projects/npp/stroke_opt_main/stroke_opt_23/results')
    parser.add_argument('--canvas_size', type=int, default=128)
    parser.add_argument('--canvas_color', type=str, default = 'black', choices=['back', 'white'])

    # Renderer settings 
    parser.add_argument('--renderer_ckpt_path', type=str, default = './model_checkpoints/renderer.pkl')
    parser.add_argument('--renderer_ckpt_path_straight', type=str, default = '/home/manuelladron/projects/npp/diff_renderer/results/checkpoints_straight_brush/renderer_250.pkl')

    # optimization settings 
    parser.add_argument('--lr', type=float, default = 0.004)

    # Stroke settings 
    parser.add_argument('--stroke_init_mode', type=str, default = 'grid', choices=['random', 'grid'])

    # loss settings 
    parser.add_argument('--w_perc', type=float, default = 0)

    # Clip loss settings 
    parser.add_argument('--add_style', type=bool, default=False)
    parser.add_argument('--opt_steps_style', type=int, default=500) # exp_17 -> 350

    parser.add_argument('--style_prompt', type=str, default = 'Starry Night by Vincent Van Gogh')
    
    parser.add_argument('--style_lambda', type=float, default=900)
    parser.add_argument('--content_lambda', type=float, default=10)
    parser.add_argument('--style_patch_lambda', type=float, default=1000)

    # Style Transfer
    parser.add_argument('--style_transfer', type=bool, default=False)
    parser.add_argument('--st_content_w', type=float, default=0.01) # 0.01 works relatively well, 0.001 works well but produces patches 
    parser.add_argument('--style_img_path', type=str, default = '/home/manuelladron/projects/npp/stroke_opt_main/images/style/goya2.jpeg')

    return parser 

if __name__ == '__main__':
    args = create_parser().parse_args()
    
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
    # Call main function and measure time 
    st = time.time()
    P.paint()
    end = time.time()
    print(f'total time: {(end-st)/60} minutes')
