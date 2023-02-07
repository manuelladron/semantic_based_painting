import time 
import sys
import argparse 
import pdb 
from src.painter import Painter 

def create_parser():
    parser = argparse.ArgumentParser(description='Stroke Optimization')

    parser.add_argument('--exp_name', type=str, default = 'exp_20_baseline')
    
    # strategy settings
    parser.add_argument('--global_loss', type=bool, default = False)
    parser.add_argument('--texturize', type=bool, default = True)
    parser.add_argument('--salient_mask', type=str, default='')

    parser.add_argument('--use_segmentation_mask', type=bool, default = False)
    parser.add_argument('--paint_by_patches', type=bool, default = True)

    parser.add_argument('--start_natural_level', type=int, default = 2)
    parser.add_argument('--patch_strategy_detail', type=str, default='natural', choices=['grid', 'natural'])

    parser.add_argument('--overlap', type=int, default=20)

    parser.add_argument('--brush_sizes', type=list, default=[0.8, 0.3, 0.1, 0.05]) 
    parser.add_argument('--budgets', type=list, default=[9, 16, 49, 64]) 
    parser.add_argument('--iter_steps', type=list, default=[250, 250, 250, 250]) 

    parser.add_argument('--number_natural_patches', type=int, default=[10, 15])  # [25, 30, 25] # [30, 50, 60] WE DID NOT DEFINE THIS IN THE PAPER 

    # misc settings 
    parser.add_argument('--upsample', type=bool, default = False)
    parser.add_argument('--aspect_ratio_downsample', type=float, default=1.5)
    parser.add_argument('--image_path', type=str, default = '/home/manuelladron/projects/npp/sigg-asia-imgs/siggraph_asia/buildings/philarm4.jpeg')
    parser.add_argument('--save_dir', type=str, default = './results')
    parser.add_argument('--canvas_size', type=int, default=128)
    parser.add_argument('--canvas_color', type=str, default = 'black', choices=['back', 'white'])
    parser.add_argument('--renderer_ckpt_path', type=str, default = './model_checkpoints/renderer.pkl')

    # optimization settings 
    parser.add_argument('--lr', type=float, default = 0.004)

    # Stroke settings 
    parser.add_argument('--stroke_init_mode', type=str, default = 'grid', choices=['random', 'grid'])

    # loss settings 
    parser.add_argument('--w_perc', type=float, default = 0)

    # Clip loss settings 

    parser.add_argument('--add_style', type=bool, default=False)
    parser.add_argument('--opt_steps_style', type=int, default=500) # exp_17 -> 350

    parser.add_argument('--style_prompt', type=str, default = 'Pop art style with reds, oranges and pinks')
    
    parser.add_argument('--style_lambda', type=float, default=800)
    parser.add_argument('--content_lambda', type=float, default=10)

    parser.add_argument('--style_patch_lambda', type=float, default=1000)


    return parser 

if __name__ == '__main__':
    args = create_parser().parse_args()
    
    # Initialize painter 
    P = Painter(args)
    # Call main function and measure time 
    st = time.time()
    P.paint()
    end = time.time()
    print(f'total time: {end-st} seconds')
