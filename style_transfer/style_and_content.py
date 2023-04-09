import torch
import torch.nn as nn
import torch.nn.functional as F


"""Loss Functions
--------------
Content Loss
~~~~~~~~~~~~

The content loss is a function that represents a weighted version of the
content distance for an individual layer. The function takes the feature
maps $F_{XL}$ of a layer $L$ in a network processing input $X$ and returns the
weighted content distance $w_{CL}.D_C^L(X,C)$ between the image $X$ and the
content image $C$. The feature maps of the content image($F_{CL}$) must be
known by the function in order to calculate the content distance. We
implement this function as a torch module with a constructor that takes
$F_{CL}$ as an input. The distance $\|F_{XL} - F_{CL}\|^2$ is the mean square error
between the two sets of feature maps, and can be computed using ``nn.MSELoss``.

We will add this content loss module directly after the convolution
layer(s) that are being used to compute the content distance. This way
each time the network is fed an input image the content losses will be
computed at the desired layers and because of auto grad, all the
gradients will be computed. Now, in order to make the content loss layer
transparent we must define a ``forward`` method that computes the content
loss and then returns the layer’s input. The computed loss is saved as a
parameter of the module.

"""


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # you need to `detach' the target content from the graph used to
        # compute the gradient in the forward pass that made it so that we don't track
        # those gradients anymore
        self.target = target.detach()


    def forward(self, input):
        # this needs to be a passthrough where you save the appropriate loss value
        self.loss = F.mse_loss(input, self.target)
        return input


"""
Style Loss
~~~~~~~~~~

The style loss module is implemented similarly to the content loss
module. It will act as a transparent layer in a
network that computes the style loss of that layer. In order to
calculate the style loss, we need to compute the gram matrix $G_{XL}$. A gram
matrix is the result of multiplying a given matrix by its transposed
matrix. In this application the given matrix is a reshaped version of
the feature maps $F_{XL}$ of a layer $L$. $F_{XL}$ is reshaped to form $\hat{F}_{XL}$, a $K$\ x\ $N$
matrix, where $K$ is the number of feature maps at layer $L$ and $N$ is the
length of any vectorized feature map $F_{XL}^k$. For example, the first line
of $\hat{F}_{XL}$ corresponds to the first vectorized feature map $F_{XL}^1$.

Finally, the gram matrix must be normalized by dividing each element by
the total number of elements in the matrix. This normalization is to
counteract the fact that $\hat{F}_{XL}$ matrices with a large $N$ dimension yield
larger values in the Gram matrix. These larger values will cause the
first layers (before pooling layers) to have a larger impact during the
gradient descent. Style features tend to be in the deeper layers of the
network so this normalization step is crucial.
"""


def gram_matrix(activations):
    a, b, c, d = activations.size()  # a=batch size(=1) , b=64, c=128, d=128 (at first layer)
    # print('a: ', a)
    # print('b: ', b)
    # print('c: ', c)
    # print('d: ', d)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = activations.view(a*b, c*d)
    # as seen in class, the Gram matrix does not depend on the spatial features
    G = torch.mm(features, features.t()) # [b, b]

    # 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    normalized_gram = G.div(a*b*c*d)
    return normalized_gram


"""Now the style loss module looks almost exactly like the content loss
module. The style distance is also computed using the mean square
error between $G_{XL}$ and $G_{SL}$.
"""


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # need to detach and cache the appropriate thing
        self.target = gram_matrix(target_feature).detach()


    def forward(self, input):
        # need to cache the appropriate loss value in self.loss
        Gram_matrix = gram_matrix(input)
        # difference between gram matrix of input and gram matrix of target
        self.loss = F.mse_loss(Gram_matrix, self.target)

        return input
