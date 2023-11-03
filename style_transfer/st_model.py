
import os
import imageio
import skimage.io as skio
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import copy
import sys
from style_transfer.utils import load_image, Normalization, device, imshow, get_image_optimizer, device, to_numpy_rgb
from style_transfer.style_and_content import ContentLoss, StyleLoss
import pdb

"""A ``Sequential`` module contains an ordered list of child modules. For
instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU, MaxPool2d,
Conv2d, ReLU…) aligned in the right order of depth. We need to add our
content loss and style loss layers immediately after the convolution
layer they are detecting. To do this we must create a new ``Sequential``
module that has content loss and style loss modules correctly inserted.
"""

# desired depth layers to compute style/content losses :
#content_layers_default = ['conv_4']
#style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# style_layers_default = ['conv_1_1', 'conv_2_1', 'conv_3_1', 'conv_4_1', 'conv_5_1']
# content_layers_default = [['conv_1_1'], ['conv_1_2'], ['conv_2_1'], ['conv_2_2'], ['conv_3_1'],
#                           ['conv_3_2'],['conv_3_3'],['conv_3_4'],['conv_4_1'], ['conv_4_2'],
#                           ['conv_4_3'],['conv_4_4'], ['conv_5_1'], ['conv_5_2'],['conv_5_3'],['conv_5_4']]

# content_layers_default = [['conv_4_4']]
# content_layers_default = [['conv_1_1']]
# style_layers_default = ['conv_1_1', 'conv_1_2', 'conv_2_1', 'conv_2_2', 'conv_3_1', 'conv_3_2', 'conv_3_3',
#                         'conv_3_4', 'conv_4_1']

# style_layers_default = ['conv_1_1', 'conv_1_2', 'conv_2_1', 'conv_2_2', 'conv_3_1']
# style_layers_default = ['conv_1_1', 'conv_1_2', 'conv_2_1']
# content_layers_default = ['conv_4_4']
# content_layers_default = ['conv_2_2']
content_layers_default = ['conv_3_1']
# style_layers_default = ['conv_1_1', 'conv_1_2', 'conv_2_1']
style_layers_default = ['conv_1_1', 'conv_1_2', 'conv_2_1', 'conv_2_2', 'conv_3_1']


# style_layers_default = [['conv_1_1'], ['conv_1_2'], ['conv_2_1'], ['conv_2_2'], ['conv_3_1'],
#                           ['conv_3_2'],['conv_3_3'],['conv_3_4'],['conv_4_1'], ['conv_4_2'],
#                           ['conv_4_3'],['conv_4_4'], ['conv_5_1'], ['conv_5_2'],['conv_5_3'],['conv_5_4']]

def get_model_and_losses(style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default, 
                               device='cuda'):
    
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # Normalization
    normalization = Normalization().to(device)

    # build a sequential model consisting of a Normalization layer
    # then all the layers of the VGG feature network along with ContentLoss and StyleLoss
    # layers in the specified places

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # here if you need a nn.ReLU layer, make sure to use inplace=False
    # as the in place version interferes with the loss layers
    # trim off the layers after the last content and style losses
    # as they are vestigial

    model = nn.Sequential(normalization)#.to(device)

    i=0 # everytime we see a conv we increment this 15 CONV LAYERS IN TOTAL
    block = 1
    conv_b = 1

    for layer in cnn.children():
        print(f'i:{i}, layer: {layer.__class__.__name__}')
        if isinstance(layer, nn.Conv2d):
            i += 1
            #name = 'conv_{}'.format(i)
            name = f'conv_{block}_{conv_b}'
            conv_b += 1
        elif isinstance(layer, nn.ReLU):
            name = 'relu_'.format(i)
            layer = nn.ReLU(inplace=False)

        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            block += 1
            conv_b = 1
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)

        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        print('\nName: ', name)

        if name in content_layers:
            target = model(content_img.float()).detach() # Target is our content image
            content_loss = ContentLoss(target)
            model.add_module('content_loss_{}'.format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img.float()).detach()
            print('shape: ', target_feature.shape)
            style_loss = StyleLoss(target_feature)
            model.add_module('style_loss_{}'.format(i), style_loss)
            style_losses.append(style_loss)

    print('\nnew model: ', model)
    # chop off the layers after last content and style losses
    for i in range(len(model) -1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i+1)]
    print('\nchopped off model: ', model)
    print('\nStyle losses: ', style_losses)
    print('\nContent losses: ', content_losses)

    return model, style_losses, content_losses


def get_model(cnn, content_layers=content_layers_default, style_layers=style_layers_default):
    
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # Normalization
    normalization = Normalization()

    # build a sequential model consisting of a Normalization layer
    # then all the layers of the VGG feature network along with ContentLoss and StyleLoss
    # layers in the specified places

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # here if you need a nn.ReLU layer, make sure to use inplace=False
    # as the in place version interferes with the loss layers
    # trim off the layers after the last content and style losses
    # as they are vestigial

    model = nn.Sequential(normalization)

    i=0 # everytime we see a conv we increment this 15 CONV LAYERS IN TOTAL
    block = 1
    conv_b = 1

    for layer in cnn.children():
        print(f'i:{i}, layer: {layer.__class__.__name__}')
        if isinstance(layer, nn.Conv2d):
            i += 1
            #name = 'conv_{}'.format(i)
            name = f'conv_{block}_{conv_b}'
            conv_b += 1
        elif isinstance(layer, nn.ReLU):
            name = 'relu_'.format(i)
            layer = nn.ReLU(inplace=False)

        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            block += 1
            conv_b = 1
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)

        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        print('\nName: ', name)

    print('\nnew model: ', model)
    # chop off the layers after last content and style losses
    for i in range(len(model) -1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i+1)]
    print('\nchopped off model: ', model)
    
    return model



"""Finally, we must define a function that performs the neural transfer. For
each iteration of the networks, it is fed an updated input and computes
new losses. We will run the ``backward`` methods of each loss module to
dynamicaly compute their gradients. The optimizer requires a “closure”
function, which reevaluates the module and returns the loss.

We still have one final constraint to address. The network may try to
optimize the input with values that exceed the 0 to 1 tensor range for
the image. We can address this by correcting the input values to be
between 0 to 1 each time the network is run.

"""
def run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True, num_steps=300,
                     style_weight=1000000, content_weight=1, content_layer = None, style_layer=None):
    """Run the image reconstruction, texture synthesis, or style transfer."""
    print('Building the style transfer model..')
    # get your model, style, and content losses
    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img,
                                                               content_layers=content_layer,
                                                               style_layers=style_layer)
    # get the optimizer
    optimizer = get_image_optimizer(input_img)
    # run model training, with one weird caveat
    # we recommend you use LBFGS, an algorithm which preconditions the gradient
    # with an approximate Hessian taken from only gradient evaluations of the function
    # this means that the optimizer might call your function multiple times per step, so as
    # to numerically approximate the derivative of the gradients (the Hessian)
    # so you need to define a function
    if use_content == False:
        print('no content')
        content_weight = 0
    if use_style == False:
        print('no style')
        style_weight = 0

    print('Optimizing...')
    run = [0]
    while run[0] <= num_steps:

        def closure():
        # here
        # which does the following:
        # clear the gradients
        # compute the loss and it's gradient
        # return the loss
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img) # this forward pass will go through each content and style modules, and calculate mse loss
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score
        optimizer.step(closure)
    # one more hint: the images must be in the range [0, 1]
    # but the optimizer doesn't know that
    # so you will need to clamp the img values to be in that range after every step
    input_img.data.clamp_(0, 1)
    # make sure to clamp once you are done

    return input_img


def main(style_img_path, content_img_path):
    #content_images = []
    # we've loaded the images for you
    print('style path: ', style_img_path)
    print('content path: ', content_img_path)

    name_content_ext = os.path.basename(content_img_path)
    name_style_ext = os.path.basename(style_img_path)
    name_content = os.path.splitext(name_content_ext)[0]
    name_style = os.path.splitext(name_style_ext)[0]


    style_img = load_image(style_img_path)
    content_img = load_image(content_img_path)
    content_img = content_img[:, :3]
    print('style img: ', style_img.shape)
    print('content img: ', content_img.shape)
    # interative MPL
    # plt.ion()

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # plot the original input image:
    plt.figure()
    imshow(style_img, title='Style Image')

    plt.figure()
    imshow(content_img, title='Content Image')

    # we load a pretrained VGG19 model from the PyTorch models library
    # but only the feature extraction part (conv layers)
    # and configure it for evaluation
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    print('\n------ CNN {VGG-19} ------- \n', cnn)

    """
    # image reconstruction
    print("Performing Image Reconstruction from white noise initialization")
    for la in content_layers_default:
        print('\n LAAAAA: ', la)
        #input_img = random noise of the size of content_img on the correct device
        input_img = torch.randn_like(content_img, device=device)
        # output = reconstruct the image from the noise
        output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=False,
                                  content_layer=la)


        print('OUTPUT: ', output.shape)
        #path = './images/results/content/' + f'{name_content}_rec_{content_layers_default[0]}.jpg'
        path = './images/results/content/' + f'{name_content}_rec_{la[0]}.jpg'

        #plt.savefig(f'{name_content}_reconstruction_{content_layers_default[0]}.jpg')
        plt.figure()
        img_rgb = to_numpy_rgb(output)
        imageio.imwrite(path, img_rgb)
        print('Saved {}'.format(path))
        imshow(output, title=f'Reconstructed Image_{la[0]}')
        #imshow(output, title=f'Reconstructed Image_{content_layers_default[0]}')

    
    # texture synthesis
    print("Performing Texture Synthesis from white noise initialization")

    # input_img = random noise of the size of content_img on the correct device
    input_img = torch.randn_like(content_img, device=device)
    # output = synthesize a texture like style_image
    output = run_optimization(cnn, content_img, style_img, input_img, use_content=False, use_style=True,
                              content_layer=content_layers_default[0],
                              style_layer=style_layers_default)

    path = './images/results/style/' + f'{name_style}_rec_{style_layers_default}_2.jpg'

    #plt.savefig(f'{name_content}_reconstruction_{content_layers_default[0]}.jpg')
    plt.figure()
    img_rgb = to_numpy_rgb(output)
    imageio.imwrite(path, img_rgb)
    print('Saved {}'.format(path))
    imshow(output, title='Synthesized Texture')
    """

    # style transfer
    print("Performing Style Transfer from style image to the content image")
    #STYLE_WEIGHT=1000000
    STYLE_WEIGHT = 0.9

    # input_img = random noise of the size of content_img on the correct device
    # input_img = torch.randn_like(content_img, device=device)
    input_img = content_img.clone()

    # output = transfer the style from the style_img to the content image
    output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True,
                              style_weight=STYLE_WEIGHT,
                              content_layer=content_layers_default,
                              style_layer=style_layers_default)

    path = './images/results/style_transfer/' + f'style_{name_style}_content_{name_content}_w_{STYLE_WEIGHT}_' \
                                                f's_{style_layers_default}_c_' \
                                                f'{content_layers_default}_input_content_hr.jpg'
    plt.figure()
    img_rgb = to_numpy_rgb(output)
    imageio.imwrite(path, img_rgb)
    imshow(output, title='style from style img to content')

    # print("Performing Style Transfer from content image initialization")
    # input_img = content_img.clone()
    # output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True)

    # plt.figure()
    imshow(output, title='Output Image from content image intialization')

    # plt.ioff()
    plt.show()


if __name__ == '__main__':
    args = sys.argv[1:3]
    main(*args)
