import os 
import pdb 
import time
import numpy 
import torch 
from models.vgg_strotss import Vgg16_Extractor
from utils import utils 
from utils import render_utils as RU
from losses import clip_loss as CL
from losses import loss_utils
import src.optimization as opt
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

class Painter():
    """ Main painter function """
    
    def __init__(self, args):

        self.args = args

        # Logger 
        self.logger = utils.init_logger(args)

        # Get ref image and canvas 
        self.src_img, self.general_canvas, self.mask = self.get_img_and_canvas()
        
        if args.texturize:
            self.general_canvas_texture = self.general_canvas.clone()
            self.all_texturized_strokes = []

        # Get perceptual network 
        self.perc = Vgg16_Extractor(space='uniform', device=device).double()

        # Set up renderer 
        self.renderer = utils.setup_renderer(args, device)
        self.num_params = 13

    def get_img_and_canvas(self): 
        """
        Opens image path and returns the reference image torch tensor and canvas. 
        Also writes list of patches and its patches limits as class atributes.
        
        :returns: torch image and torch canvas # [1, 3, H, W]
        """ 

        # Get source image and number of patches per side 
        src_img, mask, npatches_h, npatches_w, segm_mask_ids, boundaries = utils.process_img(self.args, self.args.image_path, writer=self.logger, resize_value=None) # torch tensor [1, 3, H, W]
        self.npatches_total = npatches_h * npatches_w
        self.npatches_h, self.npatches_w  = npatches_h, npatches_w
        
        self.boundaries = None
        if self.args.use_segmentation_mask:
            self.segm_mask_ids, self.boundaries = segm_mask_ids, boundaries.unsqueeze(0) # boundaries is [1, H, W] -> unsqueeze -> [1, 1, H, W]
        
            # Increase boundaries thickness
            bounds = utils.increase_boundary_thickness(boundaries.squeeze().float(), kernel_size=9, stride=1, padding=0) # [H, W]
            self.boundaries = utils.resize_tensor(bounds.unsqueeze(0), boundaries.shape[1], boundaries.shape[2]).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

            self.logger.add_image('thicker bounds', self.boundaries.squeeze(0), global_step=0)
            self.logger.add_image('original bounds', boundaries, global_step=0)
        
        
        # Create canvas 
        if self.args.canvas_color == 'black':
            canvas = torch.zeros_like(src_img).to(device) # [1, 3, H, W]

        elif self.args.canvas_color == 'white':
            canvas = torch.ones_like(src_img).to(device)  # [1, 3, H, W]

        # Get patches and patches' boundaries 
        if self.args.paint_by_patches:
            
            # For image 
            self.patches, self.patches_limits = utils.get_patches_w_overlap(self.args, src_img, npatches_h, npatches_w, writer=self.logger, name='src_img')
            
            # For mask 
            if self.args.salient_mask:
                self.mask_patches, _ = utils.get_patches_w_overlap(self.args, mask, npatches_h, npatches_w, writer=self.logger, name='salient_mask')
            
            # For segmentation boundaries 
            if self.args.use_segmentation_mask:
                self.segm_boundaries_patches, _ = utils.get_patches_w_overlap(self.args, self.boundaries, npatches_h, npatches_w, writer=self.logger, name='boundaries')

            print(f'Total patches collected: {len(self.patches_limits)}')
            print(f'Patches height: {npatches_h}, patches width: {npatches_w}')

        return src_img.to(device), canvas, mask 

    def compute_loss(self, canvas, target_patches, use_clip=False):
        """
        We do all patches together
        :param canvas: [npatches, 3, 128, 128]
        :param target_patches: [npatches, 3, 128, 128]

        :return: loss
        """
        loss = 0.0
        
        if use_clip == False:
            # L1 loss 
            l1_loss = F.l1_loss(canvas, target_patches, reduction='mean')
            loss += l1_loss
            
            # Perc loss 
            perc_loss = 0.0
            
            if self.args.w_perc > 0:
                feat_canvas = self.perc(canvas)  # [npatches, 3, 128, 128]
                feat_target = self.perc(target_patches)  # [npatches, 3, 128, 128]

                perc_loss = 0
                for j in range(len(feat_target)):
                    perc_loss -= torch.cosine_similarity(feat_canvas[j], feat_target[j], dim=1).mean()
                perc_loss /= len(feat_target)

                loss += (perc_loss * self.args.w_perc)

            return loss, l1_loss, perc_loss

        else:
            clip_loss = CL.get_clip_loss(self.args, self.args.style_prompt, canvas, target_patches, use_patch_loss=True)
            return clip_loss


    def get_reference_patches(self, mode, level, number_natural_patches, salient_mask):
        """Crops patches from source image according to the given mode: 
        :param mode: string which is either "uniform" or "natural" 
            - If uniform: patches are already cropped as a grid in get_img_and_canvas function, just return them along the already patches limits 
            - If natural: patches are cropped based on high error map regions 

        :return: target_patches, patches_limits, indices, values, mask_patches, boundaries patches, segm patches 
        """  
        if mode == 'uniform':
            if self.args.use_segmentation_mask:
                boundaries_patches = torch.cat(self.segm_boundaries_patches, dim=0).to(device)
            else:
                boundaries_patches = 0
            return torch.cat(self.patches, dim=0).to(device), self.patches_limits, 0, 0, 0, boundaries_patches # [npatches, 3, 128, 128] <- self.patches is a list with [3, 128, 128] patches 
        
        # Compute natural patches 
        else:

            # mask_patches and boundaries_patches can be None if not using them 
            target_patches, patches_limits, indices, values, mask_patches, boundaries_patches = utils.get_natural_patches(len(self.patches), self.src_img, self.general_canvas, 
                                                                                                    self.logger, level, self.mask, salient_mask=salient_mask, boundaries=self.boundaries,
                                                                                                    number_natural_patches=number_natural_patches, path=None)

            # Update npatches_total because it is set to the number of patches in the uniform grid layers 
            self.npatches_total = len(target_patches)
            
            return target_patches, patches_limits, indices, values, mask_patches, boundaries_patches


    def optimize(self, canvas, canvas_text, canvas_style, opt_steps, budget, brush_size, level, learned_strokes, total_num_patches, mode, salient_mask=False):
        """
        Optimization function called at each painting layer. Initiates a new set of strokes, paints, compute losses and optimizes strokes.

        :param canvas: previous painted canvas, if not, it will be start a blank canvas. Either None or a tensor of shape [n_patches, 3, 128, 128]
        :param canvas_text: same as canvas but for textured strokes
        :param opt_steps: optimizatino steps, integer
        :param budget: number of strokes to initialize on this layer 
        :param brush_size: thickness for brush strokes of this layer, a float
        :param level: info about this painting layer, an integer
        :param learned_strokes: previous strokes, either None, or a tensor of shape [budget, npatches, 13]
        :param mode: uniform or natural painting flag, a string
        """
        # Get patches, uniform or natural 
        target_patches, patches_limits, indices, values, mask_patches, boundaries_patches_alpha = self.get_reference_patches(mode, level, total_num_patches, salient_mask) # [npatches, 3, 128, 128] 

        # Transform boundaries alphas to boundaries color 
        if self.args.use_segmentation_mask:
            boundaries_patches = boundaries_patches_alpha * target_patches # [total_patches, 3, 128, 128] 

        # 1) If we have already learned strokes, we start the canvas painting with those strokes, otherwise is the coarsest level and we start with a black canvas
        if learned_strokes == None:
            # Black canvas 
            prev_canvas = torch.zeros(self.npatches_total, 3, 128, 128).to(device)  # [npatches, 3, H, W]
            prev_canvas_text = torch.zeros(self.npatches_total, 3, 128, 128).to(device)  # [npatches, 3, H, W]

        else:
            prev_canvas = canvas.detach()
            if canvas_text != None:
                prev_canvas_text = canvas_text.detach()
        
        # 2) Select the canvases we want to keep painting according to high error maps. Crop again because this contains previous layers 
        if mode == 'natural' and self.args.patch_strategy_detail == 'natural':
            prev_canvas = utils.crop_image(patches_limits, self.general_canvas) # [N, 3, H, W]
            if self.args.texturize:
                prev_canvas_text = utils.crop_image(patches_limits, self.general_canvas_texture)

        # init strokes 
        strokes = utils.init_strokes_patches(budget, self.args.stroke_init_mode, device, self.npatches_total) # [budget, npatches, 13]

        # Setup optimizer 
        optimizer = torch.optim.Adam([strokes], lr=self.args.lr)
        
        # Log reference image 
        self.logger.add_image(f'ref_image', self.src_img.squeeze(), global_step=0)

        # Optimization loop   
        canvas, self.general_canvas = opt.optimization_loop(self.args, self.src_img, opt_steps, target_patches, prev_canvas, strokes, budget, 
                                                            brush_size, patches_limits, self.npatches_w, level, mode, self.general_canvas, optimizer, 
                                                            self.renderer, self.num_params, self.logger, self.perc, opt_style=False)      

        # Final blending 
        self.general_canvas = utils.compose_general_canvas(self.args, canvas, mode, patches_limits, self.npatches_w, self.general_canvas, blendin=True)
        
        # Add logger 
        self.logger.add_image(f'final_canvas_base_level_{level}', img_tensor=self.general_canvas.squeeze(0),global_step=0)

        # Save base strokes to return them 
        base_strokes = strokes.clone()
        
        # this is used only once 
        if self.args.use_segmentation_mask and level == 1:
            
            print(f'\n Initiating boundaries strokes')
            boundaries_budget = 4
            boundary_strokes, boundary_indices = utils.init_strokes_boundaries(10, boundaries_budget, device, boundaries_patches_alpha, patches_limits)
            # boundary_strokes: [4, 45 (len boundary_indices), 13]
            # boundary_indices: list of integers
            # boundaries_patches: [total_num_patches, 1, 128, 128]

            optimizer_b = torch.optim.Adam([boundary_strokes], lr=self.args.lr)
            
            # To optimize only the position of strokes -> we don't care about colors here yet. 
            # We can optimize color by using the mask and apply it to the image to get borders with colors  
            prev_canvas = torch.zeros(self.npatches_total, 3, 128, 128).to(device)  # [npatches, 1, H, W]

            # args, src_img, opt_steps, target_patches, prev_canvas, strokes, budget, brush_size, patches_limits, npatches_w, level, mode, general_canvas, optimizer, renderer, num_params, logger, perc_net, global_loss=False
            canvas_boundaries, general_canvas_with_boundaries = opt.optimization_loop_boundaries(self.args, self.src_img, 400, boundaries_patches, prev_canvas, boundary_strokes, boundaries_budget, 
                                                                brush_size, patches_limits, self.npatches_w, level, mode, self.general_canvas, optimizer_b, 
                                                                self.renderer, self.num_params, self.logger, self.perc, boundary_indices, global_loss=False) 
        

            # Render boundary strokes on the general canvas:
            # boundary_strokes contain less patches than overall canvases, so we first need to select the canvases following indices 
            canvas_selected = torch.index_select(canvas, 0, torch.Tensor(boundary_indices).int().to(device))
            canvas_selected, _, _ = utils.render(canvas_selected, boundary_strokes, boundaries_budget, brush_size, self.renderer, self.num_params)

            # Recompose all canvas -
            # Generate indices that correspond to the canvases that do not have boundaries 
            total_number_of_indices = canvas.shape[0]
            indices_no_boundaries = list(set(range(total_number_of_indices))- set(boundary_indices))
            canvas = utils.merge_tensors(canvas_selected, canvas, boundary_indices, indices_no_boundaries)
            
            # Compose all canvases patches into a bigger canavs 
            self.general_canvas = utils.compose_general_canvas(self.args, canvas, mode, patches_limits, self.npatches_w, self.general_canvas, blendin=True)
            self.logger.add_image(f'final_canvas_base_with_boundaries_level_{level}', img_tensor=self.general_canvas.squeeze(0),global_step=0)


        # Once we have the strokes optimized to approx. the src_img, we can optimize for a style 
        if self.args.add_style:
            # if level > 0:
            #     brush_size += 0.2
            # Set up canvases for style: if first layer, canvas_style is the base_canvas, if not, canvas_style is stored_canvas_style
            if canvas_style == None:
                prev_canvas_style = canvas.clone().detach() 
                self.general_canvas_style = self.general_canvas.clone().detach()
            
            else:
                prev_canvas_style = canvas_style.detach()
                # 2) Select the canvases we want to keep painting according to high error maps. Crop again because this contains previous layers 
                if mode == 'natural' and self.args.patch_strategy_detail == 'natural':
                    prev_canvas_style = utils.crop_image(patches_limits, self.general_canvas_style) # [N, 3, H, W]
            
            print(f'\n Optimizing for style now')

            canvas_style, general_canvas_style = opt.optimization_loop(self.args, self.src_img, self.args.opt_steps_style, target_patches, prev_canvas_style, strokes, budget, 
                                                                brush_size, patches_limits, self.npatches_w, level, mode, self.general_canvas_style, optimizer, 
                                                                self.renderer, self.num_params, self.logger, self.perc, opt_style=True)     

            # Final blending 
            self.general_canvas_style = utils.compose_general_canvas(self.args, canvas_style, mode, patches_limits, self.npatches_w, self.general_canvas_style, blendin=True)
            
            # Add logger 
            self.logger.add_image(f'final_canvas_style_level_{level}', img_tensor=self.general_canvas_style.squeeze(0),global_step=0)

        return canvas, base_strokes, canvas_text, canvas_style


    def paint(self):
        """ Main function that calls optimize. 
        Follows a coarse to fine approach (by layers, from coarser to finer details)
        """
        learned_strokes = None
        pad_lens = None

        # Dummy variables for canvas with and without texture 
        canvas = None
        canvas_text = None
        canvas_style = None

        strokes_list = []
        canvases_text_list = [] # list with length levels with [npatches, budget, 3, 128, 128]
        all_patches_limits = []
        
        # Number of layers is directly related to number of brush sizes (and budget for each pass)
        K_painting_layers = len(self.args.brush_sizes)

        # Paint by layers 
        for k in range(K_painting_layers):  # from coarse to fine
            print(f'\n------Level: {k}-------')
        
            # higher-level settings for each layer 
            brush_size = self.args.brush_sizes[k]
            opt_steps = self.args.iter_steps[k]
            budget = self.args.budgets[k] # per patch
            st = time.time()

            if k < self.args.start_natural_level:
                mode = 'uniform'
                total_num_patches = self.npatches_total
                salient_mask = False
            else:
                mode = 'natural'
                total_num_patches = self.args.number_natural_patches[k-self.args.start_natural_level]
                
                if self.args.salient_mask != '':
                    salient_mask = True
                else:
                    salient_mask = False
            
            print(f'Brush size: {brush_size}')
            print(f'Budget: {budget}')
            print(f'Opt steps: {opt_steps}')
            print(f'Num patches: {total_num_patches}')
            print(f'Mode: {mode}\n')

            # Main function 
            canvas, strokes, canvas_text, canvas_style = self.optimize(canvas, canvas_text, canvas_style, opt_steps, budget,
                                                        brush_size, level=k, learned_strokes=learned_strokes, total_num_patches = total_num_patches, 
                                                        mode=mode, salient_mask=salient_mask)
            
            end = time.time()
            
            print(f'\n-----Level {k} time: {(end-st)/60} minutes------\n')
            learned_strokes = strokes
            strokes_list.append(strokes)
            all_patches_limits.append(self.patches_limits)







