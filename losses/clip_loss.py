import clip
from losses.imagenet_templates import imagenet_templates
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import numpy as np
import torch
from utils import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Some big pretrained models here
_clip_model = None
_preprocess = None


def _load_clip_model():
    """Lazily load the CLIP model to avoid network calls at import time."""
    global _clip_model, _preprocess
    if _clip_model is None:
        _clip_model, _preprocess = clip.load("ViT-B/32", device=device, jit=False)
        clip.model.convert_weights(_clip_model)
    return _clip_model, _preprocess

_VGG = None


def _load_vgg():
    """Lazily load VGG19 features to avoid network downloads at import time."""
    global _VGG
    if _VGG is None:
        _VGG = models.vgg19(pretrained=True).features.to(device)
        for parameter in _VGG.parameters():
            parameter.requires_grad_(False)
    return _VGG

def get_image_prior_losses(inputs_jit):
	diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
	diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
	diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
	diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

	loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
	
	return loss_var_l2

def get_features(image, model, layers=None):

	if layers is None:
		layers = {'0': 'conv1_1',  
				  '5': 'conv2_1',  
				  '10': 'conv3_1', 
				  '19': 'conv4_1', 
				  '21': 'conv4_2', 
				  '28': 'conv5_1',
				  '31': 'conv5_2'
				 }  
	features = {}
	x = image
	for name, layer in model._modules.items():
		x = layer(x)   
		if name in layers:
			features[layers[name]] = x
	
	return features

cropper = transforms.Compose([
	transforms.RandomCrop(32)
])
augment = transforms.Compose([
	transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
	transforms.Resize(224)
])

def img_denormalize(image):
	mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
	std=torch.tensor([0.229, 0.224, 0.225]).to(device)
	mean = mean.view(1,-1,1,1)
	std = std.view(1,-1,1,1)

	image = image*std +mean
	return image

def img_normalize(image):
	mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
	std=torch.tensor([0.229, 0.224, 0.225]).to(device)
	mean = mean.view(1,-1,1,1)
	std = std.view(1,-1,1,1)

	image = (image-mean)/std
	return image


def coord_setup(size=128):
	coord = torch.zeros([1, 2, size, size], device = device)
	for i in range(size):
		for j in range(size):
			coord[0, 0, i, j] = i / (size - 1.)
			coord[0, 1, i, j] = j / (size - 1.)
	return coord


def clip_normalize(image, device):
	image = F.interpolate(image,size=224,mode='bicubic')
	mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
	std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
	mean = mean.view(1,-1,1,1)
	std = std.view(1,-1,1,1)

	image = (image-mean)/std
	return image


def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
	return [template.format(text) for template in templates]


class RandomCrop2imgs(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.cropper = transforms.Compose([transforms.RandomCrop(32)])

	def __call__(self, imgs):
		return [self.cropper(img) for img in imgs]

class RandomAugment2imgs(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.augment = transforms.Compose([
						transforms.RandomPerspective(fill=0, p=0.5, distortion_scale=0.5),
						transforms.Resize(64)
					])

	def __call__(self, imgs):
		return [self.augment(img) for img in imgs]



def encode_style_prompt_clip(style_prompt):
    """
    Encodes a source and a target text prompt with CLIP encoder, returns the normalized features of both prompts 
    """
    source = 'a Photo'
    target = style_prompt #'Starry Night by Vincent Van Gogh'
    
    clip_model, _ = _load_clip_model()
    with torch.no_grad():
        # source
        template_source = compose_text_with_templates(source, imagenet_templates)
        tokens_source = clip.tokenize(template_source).to(device)
        text_source_features = clip_model.encode_text(tokens_source).detach().mean(0, keepdim=True)
        text_source_features /= text_source_features.norm(dim=-1, keepdim=True)

        # target
        template_target = compose_text_with_templates(target, imagenet_templates)
        tokens_target = clip.tokenize(template_target).to(device)
        text_target_features = clip_model.encode_text(tokens_target).detach().mean(0, keepdim=True)
        text_target_features /= text_target_features.norm(dim=-1, keepdim=True)

    return text_source_features, text_target_features


def encode_image_clip(image):
    """
    Encodes image and returns CLIP features 
    """
    clip_model, _ = _load_clip_model()
    source_features = clip_model.encode_image(clip_normalize(image, device))
    source_features /= (source_features.clone().norm(dim=-1, keepdim=True))
    
    return source_features

def compute_clipstyle_loss(source_image, painting, text_source_features, text_target_features, use_patch_loss=False):
    """
    Computes CLIP loss according to CLIPSTyler paper 
    """
    # CLIP Encode source image
    source_feats = encode_image_clip(source_image) # [bs, 512]

    # CLIP Encode painting image
    painting_feats = encode_image_clip(painting) # [bs, 512]

    # Global image direction
    glob_img_direction = (painting_feats - source_feats) # [bs, 512]

    # Text direction 
    text_direction = (text_target_features - text_source_features).repeat(painting_feats.size(0), 1) # [bs, 512]
    text_direction /= text_direction.norm(dim=-1, keepdim=True)

    # Loss global 
    loss_glob = (1 - torch.cosine_similarity(glob_img_direction, text_direction, dim=1)).mean()

    if use_patch_loss:
        num_crops = 64
        img_proc =[]
        for n in range(num_crops):
            target_crop = cropper(painting)
            target_crop = augment(target_crop)
            img_proc.append(target_crop)

        img_aug = torch.cat(img_proc, dim=0)
        clip_model, _ = _load_clip_model()
        image_features = clip_model.encode_image(clip_normalize(img_aug, device))
        image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

        source_feats_patch = source_feats.repeat(num_crops, 1)
        img_direction = (image_features-source_feats_patch)
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
        
        text_direction_patch = text_direction.repeat(num_crops, 1)
        loss_temp = (1- torch.cosine_similarity(img_direction, text_direction_patch, dim=1))
        loss_temp[loss_temp < 0.8] = 0
        loss_patch = loss_temp.mean()

        return loss_glob, loss_patch
    
    return loss_glob


def get_clip_loss(args, style_prompt, canvas, target_image, use_patch_loss):
    loss = 0
    if args.content_lambda > 0:
        vgg = _load_vgg()
        content_features = get_features(img_normalize(canvas.float()), vgg)
        target_features = get_features(img_normalize(target_image.float()), vgg)
        
        content_loss = 0
        content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

        loss += (content_loss * args.content_lambda)

    # Extract text features 
    text_source_features, text_target_features = encode_style_prompt_clip(style_prompt)

    if use_patch_loss:
        # The 2 losses yield similar losses. global loss seems to be a bit smaller than patch loss. Like 0.84 - 0.88
        glob_style_loss, patch_style_loss = compute_clipstyle_loss(target_image, canvas, text_source_features, text_target_features, use_patch_loss=use_patch_loss)
        style_loss = (glob_style_loss * args.style_lambda + patch_style_loss * args.style_patch_lambda) # vangogh was 1000.0 0.1 seems patch loss seems ok 
        # if i % 10 == 0:
        #     print(f'\niter: {i} | glob style loss: {glob_style_loss} | patch style loss: {patch_style_loss}')
        loss += style_loss
    
    else:
        style_loss = compute_clipstyle_loss(target_image, canvas, text_source_features, text_target_features, use_patch_loss=use_patch_loss)
        loss += style_loss * args.style_lambda

    return loss 