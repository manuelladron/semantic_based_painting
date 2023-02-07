import os 
import pdb 
import time
import numpy 
import torch 
import torch.nn.functional as F
from losses import clip_loss as CL


def log_losses(logger, loss_name, loss, global_step, total_steps, level):
    logger.add_scalar(f'{loss_name}_{level}', loss, global_step=global_step)
    print(f'{loss_name}: {loss.item()} | {global_step}/{total_steps} done!')


def compute_loss(args, perc_net, canvas, target_patches, use_mse = False, use_clip=False):
    """
    We do all patches together
    :param canvas: [npatches, 3, 128, 128]
    :param target_patches: [npatches, 3, 128, 128]

    :return: loss
    """
    loss = 0.0
    
    if use_clip == False:
        
        if use_mse:
            l1_loss = torch.nn.MSELoss()(canvas, target_patches)
        else:
            # L1 loss 
            l1_loss = F.l1_loss(canvas, target_patches, reduction='mean')
        loss += l1_loss
        
        # Perc loss 
        perc_loss = 0.0
        
        if args.w_perc > 0:
            feat_canvas = perc_net(canvas)  # [npatches, 3, 128, 128]
            feat_target = perc_net(target_patches)  # [npatches, 3, 128, 128]

            perc_loss = 0
            for j in range(len(feat_target)):
                perc_loss -= torch.cosine_similarity(feat_canvas[j], feat_target[j], dim=1).mean()
            perc_loss /= len(feat_target)

            loss += (perc_loss * args.w_perc)

        return loss, l1_loss, perc_loss

    else:
        clip_loss = CL.get_clip_loss(args, args.style_prompt, canvas, target_patches, use_patch_loss=True)
        return clip_loss