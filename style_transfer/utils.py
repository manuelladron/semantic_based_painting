import torch
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu, so we can test on CPU easily

loader = transforms.Compose([
    transforms.Resize((128, 128)),  # scale imported image
    # transforms.Resize((256, 256)),  # scale imported image
    # transforms.Resize((imsize, imsize)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image


def load_image(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def to_numpy_rgb(tensor):
    """
    :param tensor: [1, 3, 128, 128]
    :return: numpy rgb (128, 128, 3)
    """
    tensor = tensor.squeeze().detach().numpy()
    rgb = np.transpose(tensor, (1, 2, 0))
    return rgb

"""Additionally, VGG networks are trained on images with each channel
normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
We will use them to normalize the image before sending it into the network.
"""

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean=cnn_normalization_mean, std=cnn_normalization_std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def get_image_optimizer(input_img):
    # we recommend that you use the L-BFGS optimizer to fit the image target
    # set up an optimizer for the input image pixel values
    # make sure to specify that we need gradients for the input_image
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    return optimizer
