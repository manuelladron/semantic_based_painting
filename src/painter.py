import os 
import pdb 
import time
import json 
import numpy as np 
import torch 
from models.vgg_strotss import Vgg16_Extractor
from utils import utils 
from utils import render_utils as RU
from losses import clip_loss as CL
from losses import loss_utils
import src.optimization as opt
import torch.nn.functional as F
from utils import stroke_initialization as SI
from utils import filter_utils as FU


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

class Painter():
    """ Main painter function """
    
    def __init__(self, args):

        self.args = args

        # Logger 
        self.logger = utils.init_logger(args)

        self.segm_budget = {'sky':9, 'mountain':9, 'dirt':36, 'tree':49}
        self.segm_order = ['background','sky', 'mountain', 'building', 'pavement', 'wall', 'roof', 'floor', 'dirt', 'grass',  'window', 'tree', 'fence', 'car', 'bus', 'boat', 'bicycle', 'person']
    
        # Create dictionary that maps ids to names 
        if args.use_segmentation_mask:
            categories_json = '/home/manuelladron/projects/npp/stroke_opt_main/stroke_opt_23/utils/coco_panoptic_cat.json'
            with open(categories_json) as f:
                categories = json.load(f)
            
            self.id_to_name = {d["id"]: d["name"] for d in categories}

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
        src_img, mask, npatches_h, npatches_w, segm_mask_ids, boundaries, segm_cat_ids, seg_labels, binary_masks_list, self.segm_mask_color = utils.process_img(self.args, 
                                                                                                            self.args.image_path, writer=self.logger, 
                                                                                                            resize_value=None) # torch tensor [1, 3, H, W]
        self.npatches_total = npatches_h * npatches_w
        self.npatches_h, self.npatches_w  = npatches_h, npatches_w
        
        self.boundaries = None
        
        if self.args.use_segmentation_contours:
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
            # if self.args.salient_mask:
            #     self.mask_patches, _ = utils.get_patches_w_overlap(self.args, mask, npatches_h, npatches_w, writer=self.logger, name='salient_mask')
            
            # For segmentation boundaries 
            if self.args.use_segmentation_contours:
                self.segm_boundaries_patches, _ = utils.get_patches_w_overlap(self.args, self.boundaries, npatches_h, npatches_w, writer=self.logger, name='boundaries')

            # For segmentation masks 
            if self.args.use_segmentation_mask:
                self.list_binary_mask_patches = []
                self.list_binary_mask_names = []
                
                for i in range(len(binary_masks_list)):
                    # Get nonzero indices 
                    nz_x, nz_y = np.nonzero(segm_cat_ids * binary_masks_list[i])
                    id = segm_cat_ids[nz_x[0], nz_y[0]]
                    name = self.id_to_name[id]
                    if '-' in name:
                        name = name.split('-')[0]
                    mask = torch.from_numpy(np.expand_dims(binary_masks_list[i], axis=(0,1))).float().to(device) # [H, W] -> [1, 1, H, W]

                    bin_mask_patches, _ = utils.get_patches_w_overlap(self.args, mask, npatches_h, npatches_w, writer=self.logger, name=f'seg_mask_{name}_id_{id}')
                    self.list_binary_mask_patches.append(bin_mask_patches)
                    self.list_binary_mask_names.append(name)

                # Order masks 
                zipped_names_and_segments = utils.order_tuple_list_by_index(list(zip(self.list_binary_mask_names, self.list_binary_mask_patches)), self.segm_order)
                
                # Unzip tuples
                self.list_binary_mask_names, self.list_binary_mask_patches = zip(*zipped_names_and_segments)

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
        
        boundaries_patches, mask_patches_list = 0, []
        
        if mode == 'uniform':
            
            if self.args.use_segmentation_contours:
                boundaries_patches = torch.cat(self.segm_boundaries_patches, dim=0).to(device)

            # Iterate over segmentation binary masks and reutrn a list of patches of length number of segmentation masks 
            if self.args.use_segmentation_mask: 
                mask_patches_list = []
                for i in range(len(self.list_binary_mask_patches)): # this is a list of length number of segemntation masks. Each element in list is another list with a bunch of patches that build up the mask 
                    mask_patches = self.list_binary_mask_patches[i]
                    mask_patches = torch.cat(mask_patches, dim=0).to(device)
                    mask_patches_list.append(mask_patches)
                
            return torch.cat(self.patches, dim=0).to(device), self.patches_limits, 0, 0, 0, boundaries_patches, mask_patches_list # [npatches, 3, 128, 128] <- self.patches is a list with [3, 128, 128] patches 
        
        # Compute natural patches 
        else:
            # mask_patches and boundaries_patches can be None if not using them 
            target_patches, patches_limits, indices, values, mask_patches, boundaries_patches = utils.get_natural_patches(len(self.patches), self.src_img, self.general_canvas, 
                                                                                                    self.logger, level, self.mask, salient_mask=salient_mask, boundaries=self.boundaries,
                                                                                                    number_natural_patches=number_natural_patches, path=None)

            # Update npatches_total because it is set to the number of patches in the uniform grid layers 
            self.npatches_total = len(target_patches)
            
            return target_patches, patches_limits, indices, values, mask_patches, boundaries_patches, 0


    def optimize(self, canvas, canvas_text, canvas_style, opt_steps, budget, brush_size, level, learned_strokes, total_num_patches, mode, salient_mask=False, debug=False):
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
        target_patches, patches_limits, indices, values, _, boundaries_patches_alpha, mask_patches_list = self.get_reference_patches(mode, level, total_num_patches, salient_mask) # [npatches, 3, 128, 128] 

        # Transform boundaries alphas to boundaries color 
        if self.args.use_segmentation_contours:
            boundaries_patches = boundaries_patches_alpha * target_patches # [total_patches, 3, 128, 128] 
        
        # Get painting chunks from masks by multiplying the alpha mask by the target patch 
        if self.args.use_segmentation_mask:
            mask_color_patches_list =[]
            for i in range(len(mask_patches_list)):
                mask_color_patch = mask_patches_list[i] * target_patches # [total_patches, 3, 128, 128] 
                mask_color_patches_list.append(mask_color_patch)

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



        # Use segmentation painting after a first (level 0) uniform pass 
        if self.args.use_segmentation_mask and level == 2:
            
            # if canvas == None:
            #     canvas = prev_canvas

            canvases_segmentation = []
            canvases_names = []
            
            # Iterate over segmentation masks, independently from each other 
            for i in range(len(self.list_binary_mask_patches)):
                
                # Select mask 
                mask = self.list_binary_mask_patches[i] # list with patches 
                color_mask = mask_color_patches_list[i] # Color mask 
                name = self.list_binary_mask_names[i]
                canvases_names.append(name)

                # Set budget per segmentation mask 
                for k,v in self.segm_budget.items():
                    if k in name:
                        budget = v
                        print(f'\nFor mask: {name}, budget: {v}')
                
                print(f'\n---- Optimizing mask: {name}------\n')

                # Initialize strokes in mask 
                min_num_pixels_mask = int((128*128)*.25) # 25% of the patch should be a mask 
                strokes, mask_indices = SI.init_strokes_with_mask(min_num_pixels_mask, budget, device, mask, patches_limits, target_patches)

                # Create a new optimizer 
                optimizer_m = torch.optim.Adam([strokes], lr=self.args.lr)
                
                # Optimization loop - canvas mask is all canvases 
                canvas_mask, general_canvas_with_mask = opt.optimization_loop_boundaries(self.args, self.src_img, 300, color_mask, prev_canvas, strokes, budget, 
                                                                    brush_size, patches_limits, self.npatches_w, level, mode, self.general_canvas, optimizer_m, 
                                                                    self.renderer, self.num_params, self.logger, self.perc, mask_indices, global_loss=False, 
                                                                    name=name, mask=mask) 

                # Render strokes on the previous layer's canvases:
                # boundary_strokes contain less patches than overall canvases, so we first need to select the canvases following indices 
                canvas_selected = torch.index_select(prev_canvas, 0, torch.Tensor(mask_indices).int().to(device))

                if self.args.filter_strokes:
                    # selects only the masks that correspond to **this** segmentation part. 
                    mask = torch.cat(mask, dim=0).to(device) # [N, 3, 128, 128]
                    mask = torch.index_select(mask, 0, torch.Tensor(mask_indices).int().to(device)) # [M, 1, 128, 128] M << N (patches where the mask is present)

                    # Filter out strokes that aren't in mask 
                    valid_strokes, valid_patch_indices, valid_strokes_indices, debug_strokes = FU.filter_strokes(target_patches, strokes, mask, mask_indices, brush_size, self.renderer, self.logger, device, debug=debug)
                    
                    # FOR DEBUGGING 
                    if debug:
                        canvas_debug, _, _ = utils.render(canvas_selected, debug_strokes, budget, brush_size, self.renderer, num_params=13)
                        total_number_of_indices = prev_canvas.shape[0]
                        indices_no_boundaries = list(set(range(total_number_of_indices)) - set(mask_indices))
                        
                        all_canvas_debug = utils.merge_tensors(canvas_debug, canvas, mask_indices, indices_no_boundaries)
                        gen_canvas_debug = utils.compose_general_canvas(self.args, all_canvas_debug, 'uniform', patches_limits, self.npatches_w, self.general_canvas, blendin=True)
                        
                        self.logger.add_image(f'canvas_debug_{name}_level_{level}', img_tensor=gen_canvas_debug.squeeze(0),global_step=0)
                    
                    """
                    valid strokes: are padded strokes of the same shape as strokes 
                    valid_patch_indices: indices of patches that have a mask AND have some valid strokes
                    valid_strokes_indices: boolean tensor of shape [budget, patches] 
                    """
                    # update canvas_selected to not paint 
                    canvas_selected = utils.render_with_filter(canvas_selected, valid_strokes, valid_patch_indices, valid_strokes_indices, brush_size, mask, self.renderer)
                    
                    # Recompose all canvases - Generate indices that correspond to the canvases that do not have boundaries 
                    total_number_of_indices = prev_canvas.shape[0]
                    indices_no_boundaries = list(set(range(total_number_of_indices)) - set(valid_patch_indices))
                    
                    # Re-merge all canvases with and without masks 
                    canvas = utils.merge_tensors(canvas_selected, canvas, valid_patch_indices, indices_no_boundaries)

                else:
                    # Update now the selected canvases
                    canvas_selected, _, _ = utils.render(canvas_selected, strokes, budget, brush_size, self.renderer, self.num_params)

                    # Recompose all canvases - Generate indices that correspond to the canvases that do not have boundaries 
                    total_number_of_indices = prev_canvas.shape[0]
                    indices_no_boundaries = list(set(range(total_number_of_indices)) - set(mask_indices))
                    
                    canvas = utils.merge_tensors(canvas_selected, canvas, mask_indices, indices_no_boundaries)
                
                canvases_segmentation.append(canvas)

                # Compose all canvases patches into a bigger canavs 
                self.general_canvas = utils.compose_general_canvas(self.args, canvas, 'uniform', patches_limits, self.npatches_w, self.general_canvas, blendin=True)
                #self.logger.add_image(f'canvas_segmented_{name}_level_{level}', img_tensor=gen_canvas.squeeze(0),global_step=0)

                # Here we are updating canvas in between the masks but we should do this afterward 
                prev_canvas = canvas.detach()
                
            # Once all masks are painted, put them together 
            
            # We have canvases name and canvas segmentation in a random order. We need to asign an order to paint for instance a person after the pavement. 
            # 1) Sort the canvases_names by the established order above 
            # 2) Iterate over canvases_segmentation list based on the new order 

            #zipped_names_segments = utils.order_tuple_list_by_index(list(zip(canvases_names, canvases_segmentation)), self.segm_order)
            
            # for i in range(len(zipped_names_segments)):
            #     this_name = zipped_names_segments[i][0]
            #     this_canvas = zipped_names_segments[i][1]

            #     print('this name: ', this_name)
            #     self.general_canvas = utils.compose_general_canvas(self.args, this_canvas, 'uniform', patches_limits, self.npatches_w, self.general_canvas, blendin=True)
                
            #     self.logger.add_image(f'final_canvas_mask_{this_name}_level_{level}', img_tensor=self.general_canvas.squeeze(0),global_step=0)

            # Overlay segmentation mask with canvas 
            general_canvas_np = self.general_canvas.cpu().squeeze().permute(1,2,0).numpy() # [C,H,W] -> [H,W,C]
            segm_mask_color_np = self.segm_mask_color.cpu().permute(1,2,0).numpy() # [C,H,W] -> [H,W,C]
            
            segm_mask_overlay_canvas = utils.overlay_image_and_mask(general_canvas_np, segm_mask_color_np, alpha=0.5, img_normalized=True)
            segm_mask_overlay_canvas = torch.from_numpy(segm_mask_overlay_canvas.transpose(2,0,1)) # [C, H, W]
            
            self.logger.add_image(f'final_canvas_overlay_lvl_{level}', img_tensor=segm_mask_overlay_canvas.squeeze(0),global_step=0)
            self.logger.add_image(f'final_canvas_lvl_{level}', img_tensor=self.general_canvas.squeeze(0),global_step=0)
        
        # Uniform / natural painting without using masks 
        else:
            # init strokes
            strokes = SI.init_strokes_patches(budget, self.args.stroke_init_mode, device, self.npatches_total) # [budget, npatches, 13]

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
        if self.args.use_segmentation_contours and level == 1:
            print(f'\n------Optimizing contouring strokes-------')

            # Init strokes 
            boundaries_budget = 4
            min_num_pixels_boundary = 10
            boundary_strokes, boundary_indices = SI.init_strokes_with_mask(min_num_pixels_boundary, boundaries_budget, device, boundaries_patches_alpha, patches_limits, target_patches, edges=True)
            
            # boundaries_patches: [total_num_patches, 1, 128, 128]
            # boundary_strokes: [4, 45 (len boundary_indices), 13]
            # boundary_indices: list of integers with indices corresponding to patches that have boundaries 

            # Create a new optimizer 
            optimizer_b = torch.optim.Adam([boundary_strokes], lr=self.args.lr)
            
            # Generate canvas to optimize (we don't care about previous painting layers, this is just an empty canvas)
            prev_canvas = torch.zeros(self.npatches_total, 3, 128, 128).to(device)  # [npatches, 1, H, W]

            # Optimization loop 
            canvas_boundaries, general_canvas_with_boundaries = opt.optimization_loop_boundaries(self.args, self.src_img, 300, boundaries_patches, prev_canvas, boundary_strokes, boundaries_budget, 
                                                                brush_size, patches_limits, self.npatches_w, level, mode, self.general_canvas, optimizer_b, 
                                                                self.renderer, self.num_params, self.logger, self.perc, boundary_indices, global_loss=False) 
        

            # Render boundary strokes on the previous layer's canvases:
            # boundary_strokes contain less patches than overall canvases, so we first need to select the canvases following indices 
            canvas_selected = torch.index_select(canvas, 0, torch.Tensor(boundary_indices).int().to(device))

            # Update now the selected canvases
            canvas_selected, _, _ = utils.render(canvas_selected, boundary_strokes, boundaries_budget, brush_size, self.renderer, self.num_params)

            # Recompose all canvases - Generate indices that correspond to the canvases that do not have boundaries 
            total_number_of_indices = canvas.shape[0]
            indices_no_boundaries = list(set(range(total_number_of_indices)) - set(boundary_indices))
            
            canvas = utils.merge_tensors(canvas_selected, canvas, boundary_indices, indices_no_boundaries)
            
            # Compose all canvases patches into a bigger canavs 
            self.general_canvas = utils.compose_general_canvas(self.args, canvas, 'uniform', patches_limits, self.npatches_w, self.general_canvas, blendin=True)
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

    
        # Add texture 
        if self.args.texturize:
            canvas_text, _, _  = utils.render(prev_canvas_text, strokes, budget, brush_size, self.renderer, self.num_params, texturize=True, painter=self, writer=self.logger)
            self.general_canvas_texture = utils.compose_general_canvas(self.args, canvas_text, mode, patches_limits, self.npatches_w, self.general_canvas_texture, blendin=True)

            self.logger.add_image(f'final_canvas_texture_level_{level}', img_tensor=self.general_canvas_texture.squeeze(0),global_step=0)

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
                                                        mode=mode, salient_mask=salient_mask, debug=False)
            
            end = time.time()
            
            print(f'\n-----Level {k} time: {(end-st)/60} minutes------\n')
            learned_strokes = strokes
            strokes_list.append(strokes)
            all_patches_limits.append(self.patches_limits)






