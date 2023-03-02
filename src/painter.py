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

        self.segm_budget = {'sky':9, 'mountain':9, 'sea':49, 'rock': 9, 'dirt':81, 'tree':36, 'roof':16, 'road':36, 'person':25, 'pavement':36, 'house': 16, 'car':9, 'bicycle':9, 'backpack':4, 'potted plant': 16, 'light': 9, 'curtain':16, 'water':64, 'boat': 16 'chair':25, 'rug':9, 'dining table': 25}
        self.segm_order = ['background','sky', 'mountain', 'sea', 'rock', 'building', 'house', 'pavement', 'wall', 'roof', 'floor', 'dirt', 'road', 'water', 'grass',  'window', 'curtain', 'tree', 'fence', 'light', 'car', 'bus', 'boat', 'rug', 'bicycle', 'chair', 'dining table', 'person', 'backpack', 'potted plant']
    
        # Create dictionary that maps ids to names 
        if args.use_segmentation_mask:
            categories_json = '/home/manuelladron/projects/npp/stroke_opt_main/stroke_opt_23/utils/coco_panoptic_cat.json'
            with open(categories_json) as f:
                categories = json.load(f)
            
            self.id_to_name = {d["id"]: d["name"] for d in categories}

        if args.return_segmented_areas:
            self.all_segmentation_canvases_process = []
            if args.texturize:
                self.all_segmentation_canvases_text_process = []

        # Get ref image and canvas 
        self.src_img, self.general_canvas, self.mask = self.get_img_and_canvas()
        
        if args.texturize:
            self.general_canvas_texture = self.general_canvas.clone()
            self.all_texturized_strokes = []

        if args.return_segmented_areas:
            self.black_canvas = torch.zeros_like(self.general_canvas).to(device)

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
            self.patches, self.patches_limits, _ = utils.get_patches_w_overlap(self.args, src_img, npatches_h, npatches_w, writer=self.logger, name='src_img')
            
            # For mask 
            # if self.args.salient_mask:
            #     self.mask_patches, _, _ = utils.get_patches_w_overlap(self.args, mask, npatches_h, npatches_w, writer=self.logger, name='salient_mask')
            
            # For segmentation boundaries 
            if self.args.use_segmentation_contours:
                self.segm_boundaries_patches, _, _ = utils.get_patches_w_overlap(self.args, self.boundaries, npatches_h, npatches_w, writer=self.logger, name='boundaries')

            # For segmentation masks 
            if self.args.use_segmentation_mask:
                
                list_binary_mask_patches = [] # list with patches [M, 1, 128, 128]
                list_binary_mask_names = [] # list with names 
                list_binary_masks = [] # list with entire masks [1, 1, H, W]
                list_binary_mask_patches_with_true = []
                
                for i in range(len(binary_masks_list)):
                    # Get nonzero indices 
                    nz_x, nz_y = np.nonzero(segm_cat_ids * binary_masks_list[i])
                    id = segm_cat_ids[nz_x[0], nz_y[0]]
                    name = self.id_to_name[id]
                    if '-' in name:
                        name = name.split('-')[0]
                    mask = torch.from_numpy(np.expand_dims(binary_masks_list[i], axis=(0,1))).float().to(device) # [H, W] -> [1, 1, H, W]

                    bin_mask_patches, _, patches_with_true = utils.get_patches_w_overlap(self.args, mask, npatches_h, npatches_w, writer=self.logger, name=f'seg_mask_{name}_id_{id}', is_mask=True)
                    list_binary_mask_patches.append(bin_mask_patches)
                    list_binary_mask_names.append(name)
                    list_binary_masks.append(mask)
                    list_binary_mask_patches_with_true.append(patches_with_true)

                    # Save image 
                    basename = os.path.basename(self.args.image_path).split(".")[0]
                    img_name = f'{basename}_mask_{name}.jpg'
                    utils.save_img(mask, self.args.save_dir, img_name)

                    # Add an entire black canvas per segmentation mask to render independent segments 
                    if self.args.return_segmented_areas:
                        black_canvas = torch.zeros_like(canvas).to(device)
                        self.all_segmentation_canvases_process.append(black_canvas)
                        
                        if self.args.texturize:
                            self.all_segmentation_canvases_text_process.append(black_canvas)

                # Order masks based on a segmentation order 
                zipped_names_and_segments = utils.order_tuple_list_by_index(list(zip(list_binary_mask_names, list_binary_mask_patches, list_binary_masks, list_binary_mask_patches_with_true)), self.segm_order)
                
                # Unzip tuples
                self.list_binary_mask_names, self.list_binary_mask_patches, self.list_binary_masks, self.list_binary_mask_patches_with_true = zip(*zipped_names_and_segments)

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


    def get_reference_patches(self, mode, level, number_natural_patches):
        """Crops patches from source image according to the given mode: 
        :param mode: string which is either "uniform" or "natural" 
            - If uniform: patches are already cropped as a grid in get_img_and_canvas function, just return them along the already patches limits 
            - If natural: patches are cropped based on high error map regions 

        :return: target_patches, patches_limits, indices, values, mask_patches, boundaries patches, segm patches 
        """  
    
        # Compute uniform patches 
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

            return torch.cat(self.patches, dim=0).to(device), self.patches_limits, 0, 0, mask_patches_list, boundaries_patches # [npatches, 3, 128, 128] <- self.patches is a list with [3, 128, 128] patches 
        
        # Compute natural patches 
        else:
            # 3 ways of getting natural patches: Without segmentation masks, with segmentation masks, and with contour lines 
            
            # 1) Using segmentation masks 
            target_patches, patches_limits, indices, values, mask_patches, names = [],[],[],[],[],[]
            unvalid_indexes = []
            if self.args.use_segmentation_mask:
                
                for i in range(len(self.list_binary_masks)): # this is a list of length number of segemntation masks. Each element in list is another list with a bunch of patches that build up the mask 
                    mask = self.list_binary_masks[i] # [1, 1, H, W]
                    name = self.list_binary_mask_names[i]
                    number_natural_patches = sum(self.list_binary_mask_patches_with_true[i]) #* 2
                    
                    if number_natural_patches < 10: 
                        number_natural_patches = 10
                    
                    target_patches_m, patches_limits_m, indices_m, values_m, mask_patches_m = utils.get_natural_patches(len(self.patches), self.src_img, self.general_canvas, 
                                                                                                                self.logger, level, mask, K_number_natural_patches=number_natural_patches, 
                                                                                                                path=self.args.save_dir, name=name)
                    """
                    target_patches_m: [K, 3, 128, 128]
                    patches_limits_m: list of size K -> each element is a list with [(xmin,ymin),(xmax,ymax)]
                    indices_m: Long tensor of shape [M] with the indices of the mask 
                    values_m: Float tensor of shape [K] from high to low values 
                    mask_patches_m: tensor of shape [K, 1, 128, 128]
                    """
                    # Filter out unvalid masks either bc they are too small or not enough high error map patches
                    if isinstance(target_patches_m, int):
                        print(f'\nMask: {name}, no patches')
                        unvalid_indexes.append(i)
                        continue 
                    
                    target_patches.append(target_patches_m)
                    patches_limits.append(patches_limits_m)
                    indices.append(indices_m)
                    values.append(values_m)
                    mask_patches.append(mask_patches_m)
                    names.append(name)

                    print(f'\nMask: {name}: Number of natural patches: {target_patches_m.shape[0]}')

                # Filter out unvalid masks 
                self.list_binary_masks = utils.remove_elements_by_indexes(self.list_binary_masks, set(unvalid_indexes))
                self.list_binary_mask_names = utils.remove_elements_by_indexes(self.list_binary_mask_names, set(unvalid_indexes))
                self.list_binary_mask_patches = utils.remove_elements_by_indexes(self.list_binary_mask_patches, set(unvalid_indexes))
                
                assert len(self.list_binary_masks) == len(target_patches)
                assert len(self.list_binary_masks) == len(mask_patches)
            
            # 2) Using contour lines 
            elif self.args.use_segmentation_contours:

                target_patches, patches_limits, indices, values, mask_patches = utils.get_natural_patches(len(self.patches), self.src_img, self.general_canvas, 
                                                                                            self.logger, level, mask=self.boundaries,
                                                                                            K_number_natural_patches=number_natural_patches, path=self.args.save_dir)

            # 3) Not using masks, the entire painting is one layer 
            else:
                number_natural_patches = int(len(self.patches) * (0.7 - (level/10)))  # fewer patches at higher levels  
                target_patches, patches_limits, indices, values, mask_patches = utils.get_natural_patches(len(self.patches), self.src_img, self.general_canvas, 
                                                                                            self.logger, level, mask=None,
                                                                                            K_number_natural_patches=number_natural_patches, path=self.args.save_dir)
            
            # Update npatches_total because it is set to the number of patches in the uniform grid layers 
            self.npatches_total = len(target_patches)
            
            return target_patches, patches_limits, indices, values, mask_patches, boundaries_patches 


    def optimize(self, canvas, canvas_text, canvas_style, opt_steps, budget, brush_size, level, learned_strokes, 
                        total_num_patches, mode, debug=False, all_canvases_process=None, all_canvases_process_text=None):
        
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
        target_patches, patches_limits, indices, values, mask_patches_list, boundaries_patches_alpha = self.get_reference_patches(mode, level, total_num_patches) # [npatches, 3, 128, 128] 

        # Transform boundaries alphas to boundaries color 
        if self.args.use_segmentation_contours:
            boundaries_patches = boundaries_patches_alpha * target_patches # [M, 3, 128, 128] 
        
        # Get image parts via masks by multiplying the alpha mask by the target patch 
        if self.args.use_segmentation_mask:
            mask_color_patches_list = []
            
            for i in range(len(mask_patches_list)):
                if mode == 'uniform': # Target patches is the same for each layer, since it is not divided by layers 
                    mask_color_patch = mask_patches_list[i] * target_patches # [M, 3, 128, 128] 
                
                else: # target patches are different for each layer 
                    mask_color_patch = mask_patches_list[i] * target_patches[i] # [M, 3, 128, 128] 
                mask_color_patches_list.append(mask_color_patch)
            
        
        # 1) If no learned strokes, start new canvases, otherwise, use the previous layer canvases 
        if learned_strokes == None:
            # Black canvas 
            prev_canvas = torch.zeros(self.npatches_total, 3, 128, 128).to(device)  # [npatches, 3, H, W]
            prev_canvas_text = torch.zeros(self.npatches_total, 3, 128, 128).to(device)  # [npatches, 3, H, W]
            
        # Use canvas from previous layer 
        else:
            prev_canvas = canvas.detach()
            if canvas_text != None:
                prev_canvas_text = canvas_text.detach()


        if level <= self.args.start_using_masks and (self.args.use_segmentation_mask and self.args.return_segmented_areas):
            """
            Note: Initiating below new black patch canvases per segmentation mask, and update them after each segmentation layer to keep painting on top 
            at each of the K-th painting layers. Above in get_src_and_img function we have initiated a general_canvas (self.all_segmentation_canvases_process) 
            for each mask layer to compose the process canvases on. 
            """
            all_prev_canvases_segm = []
            all_prev_canvases_segm_text = []
            
            for i in range(len(patches_limits)):
                prev_canvas_segm_process = torch.zeros(self.npatches_total, 3, 128, 128).to(device)  # [npatches, 3, H, W]
                all_prev_canvases_segm.append(prev_canvas_segm_process)
                all_prev_canvases_segm_text.append(prev_canvas_segm_process.clone())

        elif level >= self.args.start_using_masks and (self.args.use_segmentation_mask and self.args.return_segmented_areas):
            # Same for process canvas 
            if self.args.use_segmentation_mask and self.args.return_segmented_areas:
                
                all_prev_canvases_segm = []
                all_prev_canvases_segm_text = []
                
                for i in range(len(all_canvases_process)):
                    all_prev_canvases_segm.append(all_canvases_process[i].detach())
                    
                    if self.args.texturize:
                        all_prev_canvases_segm_text.append(all_canvases_process_text[i].detach())

        
        # 2) Select the canvases we want to keep painting according to high error maps. Crop again from general canvas because this contains previous layers. 
        # The difference with uniform is that at each layer k, natural patches are different whereas uniform patches are the same all throught layers 
        if mode == 'natural' and self.args.patch_strategy_detail == 'natural':
            
            # We iterate over patches limits as they correspond to different masks 
            if self.args.use_segmentation_mask:
                all_prev_canvases = []
                all_prev_canvases_text = []
                
                all_prev_canvases_segm = []
                all_prev_canvases_segm_text = []

                # Iterate over patches limits, which is a list with a bunch of patches boundaries. A list lenght number of segmentation masks. Each segmentation mask 
                # has a different set of patches 
                for i in range(len(patches_limits)):
                    prev_canvas_m = utils.crop_image(patches_limits[i], self.general_canvas) # [M, 3, H, W]
                    all_prev_canvases.append(prev_canvas_m)
                    
                    if self.args.texturize:
                        prev_canvas_text = utils.crop_image(patches_limits[i], self.general_canvas_texture)
                        all_prev_canvases_text.append(prev_canvas_text)

                    # Do the same for the general_canvas_process 
                    if self.args.use_segmentation_mask and self.args.return_segmented_areas:
                        
                        prev_canvas_segm_m = utils.crop_image(patches_limits[i], self.all_segmentation_canvases_process[i]) # [M, 3, H, W]
                        all_prev_canvases_segm.append(prev_canvas_segm_m)

                        if self.args.texturize:
                            prev_canvas_segm_text_m = utils.crop_image(patches_limits[i], self.all_segmentation_canvases_text_process[i]) # [M, 3, H, W]
                            all_prev_canvases_segm_text.append(prev_canvas_segm_text_m)

            # Crop general_canvas based on the new patches limits that we just computed, easy
            else:
                prev_canvas = utils.crop_image(patches_limits, self.general_canvas) # [M, 3, H, W]
                if self.args.texturize:
                    prev_canvas_text = utils.crop_image(patches_limits, self.general_canvas_texture)

        # Placeholder for when we do not use segmentation masks 
        process_canvases, process_canvases_text = [], []
        
        # Use segmentation painting after a first (level 0) uniform pass 
        if self.args.use_segmentation_mask and level >= self.args.start_using_masks:
            
            canvases_segmentation = []
            canvases_names = []

            process_canvases = []
            process_canvases_text = []
            
            # Iterate over segmentation masks, independently from each other 
            for i in range(len(self.list_binary_mask_patches)):
                
                if self.args.return_segmented_areas:
                    # Black canvas for segmentation mask 
                    black_canvas = torch.zeros(self.npatches_total, 3, 128, 128).to(device)  # [npatches, 3, H, W]
                    self.black_canvas = torch.zeros_like(self.black_canvas).to(device)
                
                # Select mask 
                mask = self.list_binary_mask_patches[i] # List with patches 
                color_mask = mask_color_patches_list[i] # Color mask 
                name = self.list_binary_mask_names[i]   # Segmentation name 
                canvases_names.append(name)

                # Get previous segmentation process canvas 
                if self.args.return_segmented_areas:
                    # General 
                    general_canvas_process = self.all_segmentation_canvases_process[i] 
                    
                    # Patches 
                    prev_canvas_process = all_prev_canvases_segm[i]
                    if self.args.texturize:
                        general_canvas_process_text = self.all_segmentation_canvases_text_process[i]
                        prev_canvas_process_text = all_prev_canvases_segm_text[i]
                
                # if name == 'rock':
                #     continue 

                # Each segmentation mask is associated with a different set of target patches 
                if mode == 'natural':
                    target_patches_m = target_patches[i]
                    patches_limits_m = patches_limits[i]
                    indices_m = indices[i]
                    values_m = values[i]
                    mask = mask_patches_list_m = mask_patches_list[i] # a tensor of shape [M, 1, 128, 128]. Selected patches within mask that correspond to high error maps 
                    prev_canvas = all_prev_canvases[i] # a tensor of shape [M, 1, 128, 128]
                    prev_canvas_text = all_prev_canvases_text[i]

                    if isinstance(target_patches_m, int):
                        print(f'not enough details to paint {name}')
                        return prev_canvas, 0, 0, 0
                
                else:
                    target_patches_m = target_patches
                    patches_limits_m = patches_limits
                    indices_m = indices
                    values_m = values
                    mask_patches_list_m = mask_patches_list # list with length num_segmentation masks, and each element is [N, 1, 128, 128]

                # Set budget per segmentation mask 
                for k,v in self.segm_budget.items():
                    if k in name:
                        budget = v
                        print(f'\nFor mask: {name}, budget: {v}')
                
                print(f'\n---- Optimizing mask: {name}------\n')

                # Initialize strokes in mask and set indices corresponding to where mask == True are 
                min_num_pixels_mask = 10
                strokes, mask_indices = SI.init_strokes_with_mask(min_num_pixels_mask, budget, device, mask, patches_limits_m, target_patches_m, name=name)
                # Warning: in natural mode, mask indices is a set of indices within the highest K error maps in the mask. So these indices have little meaning in the natural appraoch 

                # Create a new optimizer 
                optimizer_m = torch.optim.Adam([strokes], lr=self.args.lr)
                
                """
                Optimization loop - canvas mask is all canvases 
                
                color_mask: [M, 3, 128, 128] result of multiplying mask patch with target patch 
                prev_canvas: [M, 3, 128, 128]
                strokes: [budget, M, 13]
                patches_limits_m: list of length M
                mask = mask_patches_list_m: [M, 1, 128, 128]
                mask_indices: a list of length M
                """    
                canvas_mask, general_canvas_with_mask = opt.optimization_loop_mask(self.args, self.src_img, 300, color_mask, prev_canvas, strokes, budget, 
                                                                    brush_size, patches_limits_m, self.npatches_w, level, mode, self.general_canvas, optimizer_m, 
                                                                    self.renderer, self.num_params, self.logger, self.perc, mask_indices, global_loss=False, 
                                                                    name=name, mask=mask) 
                
                # mask_strokes contain fewer patches than the uniform--all--canvases, so we first need to select the canvases following indices where mask == True
                if mode == 'uniform':
                    # Select canvases from previous canvases where mask == True 
                    canvas_selected = torch.index_select(prev_canvas, 0, torch.Tensor(mask_indices).int().to(device))

                else:
                    canvas_selected = canvas_mask

                # Filter out unvaid strokes that might have drawn (partially) outside the masks. 
                if self.args.filter_strokes:
                    
                    if mode == 'uniform':
                        # selects only the masks that correspond to **this** segmentation part. 
                        mask = torch.cat(mask, dim=0).to(device) # [N, 3, 128, 128]
                        mask = torch.index_select(mask, 0, torch.Tensor(mask_indices).int().to(device)) # [M, 1, 128, 128] M << N (patches where the mask is present)

                    # Filter out strokes that aren't in mask. This is nothing to do with canvases 
                    valid_strokes, valid_patch_indices, valid_strokes_indices, debug_strokes = FU.filter_strokes(target_patches_m, strokes, mask, mask_indices, 
                                                                                            brush_size, self.renderer, self.logger, device, mode, debug=debug)
                    """
                    valid strokes: are padded strokes of the same shape as strokes 
                    valid_patch_indices: indices of patches that have a mask AND have some valid strokes
                    valid_strokes_indices: boolean tensor of shape [budget, patches] 
                    """

                    # For debugging 
                    if debug:
                        canvas_debug, _, _ = utils.render(canvas_selected, debug_strokes, budget, brush_size, self.renderer, num_params=13)
                        total_number_of_indices = prev_canvas.shape[0]
                        indices_no_boundaries = list(set(range(total_number_of_indices)) - set(mask_indices))
                        
                        all_canvas_debug = utils.merge_tensors(canvas_debug, canvas, mask_indices, indices_no_boundaries)
                        gen_canvas_debug = utils.compose_general_canvas(self.args, all_canvas_debug, 'uniform', patches_limits, self.npatches_w, self.general_canvas, blendin=True)
                        
                        self.logger.add_image(f'canvas_debug_{name}_level_{level}', img_tensor=gen_canvas_debug.squeeze(0),global_step=0)

                    # These 2 lines are for when we are not returning process canvases       
                    black_canvas_patches = None
                    black_canvas_process_patches = None
                    if self.args.return_segmented_areas:
                        
                        # For isolated mask in independent levels
                        black_canvas_patches = torch.zeros_like(canvas_selected).to(device) # prev_canvas before here 
                        black_canvas_process_patches = torch.index_select(prev_canvas_process, 0, torch.Tensor(mask_indices).int().to(device))
                        
                    # Update canvas_selected to not paint strokes outside mask boundaries  
                    canvas_selected, successful, isolated_canvas, isolated_canvas_process = utils.render_with_filter(canvas_selected, valid_strokes, valid_patch_indices, 
                                                                                                                    valid_strokes_indices, brush_size, mask, self.renderer, level, 
                                                                                                                    second_canvas=black_canvas_patches, 
                                                                                                                    third_canvas=black_canvas_process_patches)
                                                                   
                    if self.args.texturize:
                        canvas_selected_text = torch.index_select(prev_canvas_text, 0, torch.Tensor(mask_indices).int().to(device)) # [M, C, H, W] M << N

                        # These 2 lines are for when we are not returning process canvases       
                        black_canvas_patches_text = None
                        black_canvas_process_patches_text = None
                        
                        if self.args.return_segmented_areas:
                            # For isolated mask in independent levels: this is part of the reason why at each level we don't accumulate the previous layer's strokes
                            black_canvas_patches_text = torch.zeros_like(canvas_selected_text).to(device)
                            black_canvas_process_patches_text = torch.index_select(prev_canvas_process_text, 0, torch.Tensor(mask_indices).int().to(device))
                            
                        canvas_selected_text, successful_text, isolated_canvas_text, isolated_canvas_process_text = utils.render_with_filter(canvas_selected_text, valid_strokes, valid_patch_indices, valid_strokes_indices, brush_size, 
                                                                                                    mask, self.renderer, level, writer=self.logger, texturize=True, painter=self, segm_name=name, 
                                                                                                    second_canvas=black_canvas_patches_text, third_canvas=black_canvas_process_patches_text)
                    
                    # Means there is at least one good stroke 
                    if successful:
                        
                        # Recompose all canvases - Generate indices that correspond to the canvases that do not have boundaries 
                        total_number_of_indices = prev_canvas.shape[0]
                        indices_no_boundaries = list(set(range(total_number_of_indices)) - set(valid_patch_indices))
                        
                        # Re-merge all canvases with and without masks 
                        if mode == 'uniform':
                            canvas = utils.merge_tensors(canvas_selected, canvas, valid_patch_indices, indices_no_boundaries)
                            
                            if self.args.return_segmented_areas:
                                isolated_mask = utils.merge_tensors(isolated_canvas, black_canvas, valid_patch_indices, indices_no_boundaries)
                                process_mask = utils.merge_tensors(isolated_canvas_process, prev_canvas_process, valid_patch_indices, indices_no_boundaries)
                            
                            if self.args.texturize:
                                canvas_text = utils.merge_tensors(canvas_selected_text, canvas_text, valid_patch_indices, indices_no_boundaries)
                                if self.args.return_segmented_areas:
                                    isolated_mask_text = utils.merge_tensors(isolated_canvas_text, black_canvas, valid_patch_indices, indices_no_boundaries)
                                    process_mask_text = utils.merge_tensors(isolated_canvas_process_text, prev_canvas_process_text, valid_patch_indices, indices_no_boundaries)
                        
                        else:
                            canvas = canvas_selected
                            isolated_mask = isolated_canvas # = canvas
                            process_mask = isolated_canvas_process
                            
                            if self.args.texturize:
                                isolated_mask_text = isolated_canvas_text
                                process_mask_text = isolated_canvas_process_text
                    
                    else:
                        print('\nUnsuccessful filtering ')
                        canvas = canvas_selected

                else:
                    # Update now the selected canvases
                    canvas_selected, _, _ = utils.render(canvas_selected, strokes, budget, brush_size, self.renderer, level, self.num_params)

                    if mode == 'uniform':
                        # Recompose all canvases - Generate indices that correspond to the canvases that do not have boundaries 
                        total_number_of_indices = prev_canvas.shape[0]
                        indices_no_boundaries = list(set(range(total_number_of_indices)) - set(mask_indices))
                        canvas = utils.merge_tensors(canvas_selected, canvas, mask_indices, indices_no_boundaries)

                        if self.args.texturize:
                            assert prev_canvas.shape[0] == prev_canvas_text.shape[0]
                            
                            canvas_selected_text = torch.index_select(prev_canvas_text, 0, torch.Tensor(mask_indices).int().to(device)) # [M, C, H, W] M << N
                            
                            # Render strokes, which contain only patches that have masks or boundaries, so we need to select first only the canvases that have boundaries/masks 
                            canvas_selected_text, _, _ = utils.render(canvas_selected_text, strokes, budget, brush_size, self.renderer, self.num_params, level, texturize=True, painter=self, writer=self.logger, segm_name=name)
                            
                            # Merge all canvases (mask and no mask) - update canvas variable 
                            canvas_text = utils.merge_tensors(canvas_selected_text, canvas_text, mask_indices, sorted(indices_no_boundaries)) # [N, C, H, W]

                    else:
                        canvas = canvas_selected
                
                canvases_segmentation.append(canvas)

                # Compose all canvases patches into a bigger canavs 
                if mode == 'natural':
                    self.general_canvas = utils.compose_general_canvas(self.args, canvas, mode, patches_limits_m, self.npatches_w, self.general_canvas, blendin=True)
                    
                    if self.args.return_segmented_areas:
                        general_isolated_mask = utils.compose_general_canvas(self.args, isolated_mask, mode, patches_limits_m, self.npatches_w, self.black_canvas, blendin=True)
                        general_canvas_process = utils.compose_general_canvas(self.args, process_mask, mode, patches_limits_m, self.npatches_w, general_canvas_process, blendin=True)
                
                else:
                    self.general_canvas = utils.compose_general_canvas(self.args, canvas, mode, patches_limits, self.npatches_w, self.general_canvas, blendin=True)
                    
                    if self.args.return_segmented_areas:
                        general_isolated_mask = utils.compose_general_canvas(self.args, isolated_mask, mode, patches_limits, self.npatches_w, self.black_canvas, blendin=True)
                        general_canvas_process = utils.compose_general_canvas(self.args, process_mask, mode, patches_limits, self.npatches_w, general_canvas_process, blendin=True)
                
                # Update canvas in between the masks but we should do this afterward 
                prev_canvas = canvas.detach()

                # Update process canvas 
                if self.args.return_segmented_areas:
                    prev_canvas_process = process_mask.detach()

                    self.all_segmentation_canvases_process[i] = general_canvas_process

                    self.logger.add_image(f'final_isol_mask_{name}_lvl_{level}', img_tensor=general_isolated_mask.squeeze(0),global_step=0)
                    self.logger.add_image(f'final_process_mask_{name}_lvl_{level}', img_tensor=general_canvas_process.squeeze(0),global_step=0)
                    basename = os.path.basename(self.args.image_path).split(".")[0]
                    
                    # Independent isolated mask 
                    img_name = f'{basename}_isol_mask_{name}_lvl_{level}.jpg'
                    utils.save_img(general_isolated_mask.squeeze(), self.args.save_dir, img_name)

                    # Process mask 
                    img_name = f'{basename}_process_mask_{name}_lvl_{level}.jpg'
                    utils.save_img(general_canvas_process.squeeze(), self.args.save_dir, img_name)   


                # Texturize each mask 
                if self.args.texturize:
                    
                    if mode == 'natural':
                        canvas_text, _, _  = utils.render(prev_canvas_text, strokes, budget, brush_size, self.renderer, self.num_params, level, texturize=True, painter=self, writer=self.logger, segm_name=name)
                        self.general_canvas_texture = utils.compose_general_canvas(self.args, canvas_text, mode, patches_limits_m, self.npatches_w, self.general_canvas_texture, blendin=True)
                       
                        if self.args.return_segmented_areas:
                            general_isolated_mask_text = utils.compose_general_canvas(self.args, isolated_mask_text, mode, patches_limits_m, self.npatches_w, self.black_canvas, blendin=True)
                            general_canvas_process_text = utils.compose_general_canvas(self.args, process_mask_text, mode, patches_limits_m, self.npatches_w, general_canvas_process_text, blendin=True)

                    else:   
                        self.general_canvas_texture = utils.compose_general_canvas(self.args, canvas_text, mode, patches_limits, self.npatches_w, self.general_canvas_texture, blendin=True)
                        
                        if self.args.return_segmented_areas:
                            general_isolated_mask_text = utils.compose_general_canvas(self.args, isolated_mask_text, mode, patches_limits, self.npatches_w, self.black_canvas, blendin=True)
                            general_canvas_process_text = utils.compose_general_canvas(self.args, process_mask_text, mode, patches_limits, self.npatches_w, general_canvas_process_text, blendin=True)
                    
                    # Log canvas 
                    print('logging canvas texture lvl: ', level)
                    self.logger.add_image(f'final_canvas_texture_mask_{name}_lvl_{level}', img_tensor=self.general_canvas_texture.squeeze(0),global_step=0)
                    
                    if self.args.return_segmented_areas:
                        self.logger.add_image(f'final_isol_text_mask_{name}_lvl_{level}', img_tensor=general_isolated_mask_text.squeeze(0),global_step=0)
                        self.logger.add_image(f'final_process_text_mask_{name}_lvl_{level}', img_tensor=general_canvas_process_text.squeeze(0),global_step=0)
                        
                        # isolated 
                        img_name = f'{basename}_isol_text_mask_{name}_lvl_{level}.jpg'
                        utils.save_img(general_isolated_mask_text.squeeze(), self.args.save_dir, img_name)

                        # process 
                        img_name = f'{basename}_process_text_mask_{name}_lvl_{level}.jpg'
                        utils.save_img(general_canvas_process_text.squeeze(), self.args.save_dir, img_name)

                        self.all_segmentation_canvases_text_process[i] = general_canvas_process
                    
                    # Update canvases 
                    prev_canvas_text = canvas_text.detach()
                    if self.args.return_segmented_areas:
                        prev_canvas_process_text = process_mask_text.detach()

                if self.args.return_segmented_areas:
                    process_canvases.append(process_mask)
                    if self.args.texturize:
                        process_canvases_text.append(process_mask_text)
            
            # Overlay segmentation mask with canvas 
            general_canvas_np = self.general_canvas.cpu().squeeze().permute(1,2,0).numpy() # [C,H,W] -> [H,W,C]
            segm_mask_color_np = self.segm_mask_color.cpu().permute(1,2,0).numpy() # [C,H,W] -> [H,W,C]
            
            segm_mask_overlay_canvas = utils.overlay_image_and_mask(general_canvas_np, segm_mask_color_np, alpha=0.5, img_normalized=True)
            segm_mask_overlay_canvas = torch.from_numpy(segm_mask_overlay_canvas.transpose(2,0,1)) # [C, H, W]
            
            self.logger.add_image(f'final_canvas_lvl_{level}', img_tensor=self.general_canvas.squeeze(0),global_step=0)
            self.logger.add_image(f'final_canvas_overlay_lvl_{level}', img_tensor=segm_mask_overlay_canvas.squeeze(0),global_step=0)
            
            if self.args.texturize:
                #canvas_text, _, _  = utils.render(prev_canvas_text, strokes, budget, brush_size, self.renderer, self.num_params, texturize=True, painter=self, writer=self.logger)
                if mode == 'natural':
                    self.general_canvas_texture = utils.compose_general_canvas(self.args, canvas_text, mode, patches_limits_m, self.npatches_w, self.general_canvas_texture, blendin=True)
                else:   
                    self.general_canvas_texture = utils.compose_general_canvas(self.args, canvas_text, mode, patches_limits, self.npatches_w, self.general_canvas_texture, blendin=True)
                # Log canvas 
                print('logging canvas texture lvl: ', level)
                self.logger.add_image(f'final_canvas_texture_lvl_{level}', img_tensor=self.general_canvas_texture.squeeze(0),global_step=0)

        
        # Uniform / natural painting without using masks 
        else:
            # Init strokes
            strokes = SI.init_strokes_patches(budget, self.args.stroke_init_mode, device, self.npatches_total, target_patches) # [budget, npatches, 13]

            # Setup optimizer 
            optimizer = torch.optim.Adam([strokes], lr=self.args.lr)
            
            # Log reference image 
            self.logger.add_image(f'ref_image', self.src_img.squeeze(), global_step=0)

            # Optimization loop   
            canvas, self.general_canvas, strokes = opt.optimization_loop(self.args, self.src_img, opt_steps, target_patches, prev_canvas, strokes, budget, 
                                                                brush_size, patches_limits, self.npatches_w, level, mode, self.general_canvas, optimizer, 
                                                                self.renderer, self.num_params, self.logger, self.perc, opt_style=False)      

            # Final blending 
            self.general_canvas = utils.compose_general_canvas(self.args, canvas, mode, patches_limits, self.npatches_w, self.general_canvas, blendin=True)
            
            # Add logger 
            self.logger.add_image(f'final_canvas_base_level_{level}', img_tensor=self.general_canvas.squeeze(0),global_step=0)

            
            if self.args.texturize:
                canvas_text, _, _  = utils.render(prev_canvas_text, strokes, budget, brush_size, self.renderer, self.num_params, level, texturize=True, painter=self, writer=self.logger)
                self.general_canvas_texture = utils.compose_general_canvas(self.args, canvas_text, mode, patches_limits, self.npatches_w, self.general_canvas_texture, blendin=True)

                # Log canvas 
                print('logging canvas texture lvl: ', level)
                self.logger.add_image(f'final_canvas_texture_inloop_lvl_{level}', img_tensor=self.general_canvas_texture.squeeze(0),global_step=0)

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
            canvas_boundaries, general_canvas_with_boundaries = opt.optimization_loop_mask(self.args, self.src_img, 300, boundaries_patches, prev_canvas, boundary_strokes, boundaries_budget, 
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

            canvas_style, general_canvas_style, strokes = opt.optimization_loop(self.args, self.src_img, self.args.opt_steps_style, target_patches, prev_canvas_style, strokes, budget, 
                                                                brush_size, patches_limits, self.npatches_w, level, mode, self.general_canvas_style, optimizer, 
                                                                self.renderer, self.num_params, self.logger, self.perc, opt_style=True)     

            # Final blending 
            self.general_canvas_style = utils.compose_general_canvas(self.args, canvas_style, mode, patches_limits, self.npatches_w, self.general_canvas_style, blendin=True)
            
            # Add logger 
            self.logger.add_image(f'final_canvas_style_level_{level}', img_tensor=self.general_canvas_style.squeeze(0),global_step=0)
    
        # Save image 
        basename = os.path.basename(self.args.image_path).split(".")[0]
        img_name = f'{basename}_lvl_{level}.jpg'
        utils.save_img(self.general_canvas.squeeze(), self.args.save_dir, img_name)
        
        if level == 0:
            img_name = f'{basename}_src_img.jpg'
            utils.save_img(self.src_img.squeeze(), self.args.save_dir, img_name)
        
        if self.args.texturize:
            img_name = f'{basename}_texture_lvl_{level}.jpg'
            utils.save_img(self.general_canvas_texture.squeeze(), self.args.save_dir, img_name)

        if self.args.use_segmentation_mask and level >= self.args.start_using_masks:
            img_name = f'{basename}_mask_overlay.jpg'
            utils.save_img(segm_mask_overlay_canvas/255, self.args.save_dir, img_name)

        return canvas, strokes, canvas_text, canvas_style, process_canvases, process_canvases_text


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

        # for segmentation process 
        all_canvases_process = []
        all_canvases_process_text = []
        
        # Number of layers is directly related to number of brush sizes (and budget for each pass)
        K_painting_layers = len(self.args.brush_sizes)

        # Paint by layers 
        mode = 'uniform'
        for k in range(K_painting_layers):  # from coarse to fine
            
            print(f'\n------Level: {k}-------')
        
            # higher-level settings for each layer 
            brush_size = self.args.brush_sizes[k]
            opt_steps = self.args.iter_steps[k]
            budget = self.args.budgets[k] # per patch
            st = time.time()

            if k < self.args.start_natural_level or self.args.patch_strategy_detail == 'uniform':
                mode = 'uniform'
                total_num_patches = self.npatches_total
                salient_mask = False
            
            else:
                mode = 'natural'
                total_num_patches = self.args.number_natural_patches[k-self.args.start_natural_level]
                            
            print(f'Brush size: {brush_size}')
            print(f'Budget: {budget}')
            print(f'Opt steps: {opt_steps}')
            print(f'Num patches: {total_num_patches}')
            print(f'Mode: {mode}\n')

            # Main function 
            canvas, strokes, canvas_text, canvas_style, all_canvases_process, all_canvases_process_text = self.optimize(canvas, canvas_text, canvas_style, opt_steps, budget,
                                                        brush_size, level=k, learned_strokes=learned_strokes, total_num_patches = total_num_patches, 
                                                        mode=mode, debug=False, all_canvases_process=all_canvases_process, all_canvases_process_text=all_canvases_process_text)
            
            
            end = time.time()
            
            print(f'\n-----Level {k} time: {(end-st)/60} minutes------\n')
            learned_strokes = strokes
            strokes_list.append(strokes)
            all_patches_limits.append(self.patches_limits)







