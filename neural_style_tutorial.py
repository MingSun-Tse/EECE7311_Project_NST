"""
Neural Transfer Using PyTorch
=============================


**Author**: `Alexis Jacq <https://alexis-jacq.github.io>`_
 
**Edited by**: `Winston Herring <https://github.com/winston6>`_

Introduction
------------

This tutorial explains how to implement the `Neural-Style algorithm <https://arxiv.org/abs/1508.06576>`__
developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.
Neural-Style, or Neural-Transfer, allows you to take an image and
reproduce it with a new artistic style. The algorithm takes three images,
an input image, a content-image, and a style-image, and changes the input 
to resemble the content of the content-image and the artistic style of the style-image.

 
.. figure:: /_static/img/neural-style/neuralstyle.png
   :alt: content1
"""

######################################################################
# Underlying Principle
# --------------------
# 
# The principle is simple: we define two distances, one for the content
# (:math:`D_C`) and one for the style (:math:`D_S`). :math:`D_C` measures how different the content
# is between two images while :math:`D_S` measures how different the style is
# between two images. Then, we take a third image, the input, and
# transform it to minimize both its content-distance with the
# content-image and its style-distance with the style-image. Now we can
# import the necessary packages and begin the neural transfer.
# 
# Importing Packages and Selecting a Device
# -----------------------------------------
# Below is a  list of the packages needed to implement the neural transfer.
#
# -  ``torch``, ``torch.nn``, ``numpy`` (indispensables packages for
#    neural networks with PyTorch)
# -  ``torch.optim`` (efficient gradient descents)
# -  ``PIL``, ``PIL.Image``, ``matplotlib.pyplot`` (load and display
#    images)
# -  ``torchvision.transforms`` (transform PIL images into tensors)
# -  ``torchvision.models`` (train or load pre-trained models)
# -  ``copy`` (to deep copy the models; system package)

from __future__ import print_function
import os
import sys
import argparse
pjoin = os.path.join
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter # set ytick precision
plt.style.use(['science', 'no-latex'])

import torchvision.transforms as transforms
import torchvision.models as models
import time
import copy
from logger import Logger
from torch.autograd import Variable
import torch.fft
from utils import visualize_filter, visualize_feature_map
from collections import OrderedDict


######################################################################
# Next, we need to choose which device to run the network on and import the
# content and style images. Running the neural transfer algorithm on large
# images takes longer and will go much faster when running on a GPU. We can
# use ``torch.cuda.is_available()`` to detect if there is a GPU available.
# Next, we set the ``torch.device`` for use throughout the tutorial. Also the ``.to(device)``
# method is used to move tensors or modules to a desired device. 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# Loading the Images
# ------------------
#
# Now we will import the style and content images. The original PIL images have values between 0 and 255, but when
# transformed into torch tensors, their values are converted to be between
# 0 and 1. The images also need to be resized to have the same dimensions.
# An important detail to note is that neural networks from the
# torch library are trained with tensor values ranging from 0 to 1. If you
# try to feed the networks with 0 to 255 tensor images, then the activated
# feature maps will be unable sense the intended content and style.
# However, pre-trained networks from the Caffe library are trained with 0
# to 255 tensor images. 
#
#
# .. Note::
#     Here are links to download the images required to run the tutorial:
#     `picasso.jpg <http://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg>`__ and
#     `dancing.jpg <http://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg>`__.
#     Download these two images and add them to a directory
#     with name ``images`` in your current working directory.

def image_loader(image_name, imsize):
  loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor
    
  image = Image.open(image_name).convert("RGB")
  # fake batch dimension required to fit network's input dimensions
  image = loader(image).unsqueeze(0)
  return image.to(device, torch.float)

######################################################################
# Now, let's create a function that displays an image by reconverting a 
# copy of it to PIL format and displaying the copy using 
# ``plt.imshow``. We will try displaying the content and style images 
# to ensure they were imported correctly.

unloader = transforms.ToPILImage()  # reconvert into PIL image
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
      plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

######################################################################
# Loss Functions
# --------------
# Content Loss
# ~~~~~~~~~~~~
# 
# The content loss is a function that represents a weighted version of the
# content distance for an individual layer. The function takes the feature
# maps :math:`F_{XL}` of a layer :math:`L` in a network processing input :math:`X` and returns the
# weighted content distance :math:`w_{CL}.D_C^L(X,C)` between the image :math:`X` and the
# content image :math:`C`. The feature maps of the content image(:math:`F_{CL}`) must be
# known by the function in order to calculate the content distance. We
# implement this function as a torch module with a constructor that takes
# :math:`F_{CL}` as an input. The distance :math:`\|F_{XL} - F_{CL}\|^2` is the mean square error
# between the two sets of feature maps, and can be computed using ``nn.MSELoss``.
# 
# We will add this content loss module directly after the convolution
# layer(s) that are being used to compute the content distance. This way
# each time the network is fed an input image the content losses will be
# computed at the desired layers and because of auto grad, all the
# gradients will be computed. Now, in order to make the content loss layer
# transparent we must define a ``forward`` method that computes the content
# loss and then returns the layer’s input. The computed loss is saved as a
# parameter of the module.
# 

class ContentLoss(nn.Module):
  def __init__(self, target):
      super(ContentLoss, self).__init__()
      # we 'detach' the target content from the tree used
      # to dynamically compute the gradient: this is a stated value,
      # not a variable. Otherwise the forward method of the criterion
      # will throw an error.
      self.target = target.detach()

  def forward(self, input):
      self.loss = F.mse_loss(input, self.target)
      self.feat = input.clone().detach()
      return input

######################################################################
# .. Note::
#    **Important detail**: although this module is named ``ContentLoss``, it
#    is not a true PyTorch Loss function. If you want to define your content
#    loss as a PyTorch Loss function, you have to create a PyTorch autograd function 
#    to recompute/implement the gradient manually in the ``backward``
#    method.

######################################################################
# Style Loss
# ~~~~~~~~~~
# 
# The style loss module is implemented similarly to the content loss
# module. It will act as a transparent layer in a
# network that computes the style loss of that layer. In order to
# calculate the style loss, we need to compute the gram matrix :math:`G_{XL}`. A gram
# matrix is the result of multiplying a given matrix by its transposed
# matrix. In this application the given matrix is a reshaped version of
# the feature maps :math:`F_{XL}` of a layer :math:`L`. :math:`F_{XL}` is reshaped to form :math:`\hat{F}_{XL}`, a :math:`K`\ x\ :math:`N`
# matrix, where :math:`K` is the number of feature maps at layer :math:`L` and :math:`N` is the
# length of any vectorized feature map :math:`F_{XL}^k`. For example, the first line
# of :math:`\hat{F}_{XL}` corresponds to the first vectorized feature map :math:`F_{XL}^1`.
# 
# Finally, the gram matrix must be normalized by dividing each element by
# the total number of elements in the matrix. This normalization is to
# counteract the fact that :math:`\hat{F}_{XL}` matrices with a large :math:`N` dimension yield
# larger values in the Gram matrix. These larger values will cause the
# first layers (before pooling layers) to have a larger impact during the
# gradient descent. Style features tend to be in the deeper layers of the
# network so this normalization step is crucial.
# 

def gram_matrix(input):
  a, b, c, d = input.size()  # a=batch size(=1)
  # b=number of feature maps
  # (c,d)=dimensions of a f. map (N=c*d)

  features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

  G = torch.mm(features, features.t())  # compute the gram product

  # we 'normalize' the values of the gram matrix
  # by dividing by the number of element in each feature maps.
  return G.div(a * b * c * d)


######################################################################
# Now the style loss module looks almost exactly like the content loss
# module. The style distance is also computed using the mean square
# error between :math:`G_{XL}` and :math:`G_{SL}`.
# 

class StyleLoss(nn.Module):
  def __init__(self, target_feature):
      super(StyleLoss, self).__init__()
      self.target = gram_matrix(target_feature).detach()

  def forward(self, input):
      G = gram_matrix(input)
      self.loss = F.mse_loss(G, self.target)
      self.gram = G.clone().detach()
      self.feat = input.clone().detach()
      return input


######################################################################
# Importing the Model
# -------------------
# 
# Now we need to import a pre-trained neural network. We will use a 19
# layer VGG network like the one used in the paper.
# 
# PyTorch’s implementation of VGG is a module divided into two child
# ``Sequential`` modules: ``features`` (containing convolution and pooling layers),
# and ``classifier`` (containing fully connected layers). We will use the
# ``features`` module because we need the output of the individual
# convolution layers to measure content and style loss. Some layers have
# different behavior during training than evaluation, so we must set the
# network to evaluation mode using ``.eval()``.


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
  def __init__(self, mean, std):
      super(Normalization, self).__init__()
      # .view the mean and std to make them [C x 1 x 1] so that they can
      # directly work with image Tensor of shape [B x C x H x W].
      # B is batch size. C is number of channels. H is height and W is width.
      self.mean = mean.clone().detach().view(-1, 1, 1)
      self.std = std.clone().detach().view(-1, 1, 1)

  def forward(self, img):
      # normalize img
      return (img - self.mean) / self.std


######################################################################
# A ``Sequential`` module contains an ordered list of child modules. For
# instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU, MaxPool2d,
# Conv2d, ReLU…) aligned in the right order of depth. We need to add our
# content loss and style loss layers immediately after the convolution
# layer they are detecting. To do this we must create a new ``Sequential``
# module that has content loss and style loss modules correctly inserted.
#
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, args, content_layers, style_layers):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = OrderedDict()
    style_losses   = OrderedDict()

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        elif isinstance(layer, nn.ReflectionPad2d):
            name = 'pad_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses[name] = content_loss
        
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses[name] = style_loss
 
    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
      if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
        break

    model = model[:(i + 1)]
    netprint('New network arch:', model)
    return model, style_losses, content_losses


# This FFT image paramaterization refers to Google
# at: https://github.com/tensorflow/lucid/blob/68b826048a9edd73815283fa1016e4e1e03e7f32/lucid/optvis/param/spatial.py#L48
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""

    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def fft_image(shape, sd=None, decay_power=1):
    """An image paramaterization using 2D Fourier coefficients."""
    sd = sd or 0.01
    # batch, h, w, ch = shape # tf style: [N, H, W, C]
    batch, ch, h, w = shape # torch style: [N, C, H, W]
    
    freqs = rfft2d_freqs(h, w)
    init_val_size = (2, batch, ch) + freqs.shape

    init_val = np.random.normal(size=init_val_size, scale=sd).astype(np.float32)
    # spectrum_real_imag_t = tf.Variable(init_val)
    spectrum_real_imag_t = Variable(torch.from_numpy(init_val))

    # spectrum_t = tf.complex(spectrum_real_imag_t[0], spectrum_real_imag_t[1])
    spectrum_t = torch.complex(spectrum_real_imag_t[0], spectrum_real_imag_t[1])

    # Scale the spectrum. First normalize energy, then scale by the square-root
    # of the number of pixels to get a unitary transformation.
    # This allows to use similar leanring rates to pixel-wise optimisation.
    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale *= np.sqrt(w * h)
    print(scale.shape, spectrum_t.shape)
    scaled_spectrum_t = torch.from_numpy(scale) * spectrum_t

    # convert complex scaled spectrum to shape (h, w, ch) image tensor
    # needs to transpose because irfft2d returns channels first
    # image_t = tf.transpose(tf.spectral.irfft2d(scaled_spectrum_t), (0, 2, 3, 1))
    image_t = torch.fft.irfft(scaled_spectrum_t) # shape: [N, C, H, W]

    # in case of odd spatial input dimensions we need to crop
    # image_t = image_t[:batch, :h, :w, :ch] # tf style
    image_t = image_t[:batch, :ch, :h, :w] # torch style
    image_t = image_t / 4.0  # TODO: is that a magic constant?
    return image_t


######################################################################
# Next, we select the input image. You can use a copy of the content image
# or white noise.
# 

# input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
# plt.figure()
# imshow(input_img, title='Input Image')


######################################################################
# Gradient Descent
# ----------------
# 
# As Leon Gatys, the author of the algorithm, suggested `here <https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq>`__, we will use
# L-BFGS algorithm to run our gradient descent. Unlike training a network,
# we want to train the input image in order to minimise the content/style
# losses. We will create a PyTorch L-BFGS optimizer ``optim.LBFGS`` and pass
# our image to it as the tensor to optimize.
# 

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


######################################################################
# Finally, we must define a function that performs the neural transfer. For
# each iteration of the networks, it is fed an updated input and computes
# new losses. We will run the ``backward`` methods of each loss module to
# dynamicaly compute their gradients. The optimizer requires a “closure”
# function, which reevaluates the modul and returns the loss.
# 
# We still have one final constraint to address. The network may try to
# optimize the input with values that exceed the 0 to 1 tensor range for
# the image. We can address this by correcting the input values to be
# between 0 to 1 each time the network is run.
# 

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       args, content_layers_default, style_layers_default, num_steps,
                       style_weight, content_weight): # default: style_weight = 1e6, content_weight = 1
    """Run the style transfer."""
    content_img = image_loader(args.content, args.img_size)
    style_img   = image_loader(args.style, args.img_size)
    input_img   = content_img.clone()
    assert style_img.size() == content_img.size(), \
      "we need to import style and content images of the same size"
      
    logprint('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img, 
        args, content_layers_default, style_layers_default)
    
    if args.fft:
        input_img = fft_image(input_img.shape).to(device, torch.float) # convert to fft parameterization
    optimizer = get_input_optimizer(input_img)
    
    logprint('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1) # correct the values of updated input image
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for layer_name, sl in style_losses.items():
                style_score += sl.loss
                if args.plot_feature and run[0] == num_steps: # visualize feature maps at the last iter
                    analyze_gram(sl.gram, layer_name) # analyze the gram matrix, like SVD analysis
                    visualize_feature_map(sl.feat, layer_id=layer_name, save_dir=logger.gen_img_path, prefix=prefix, ext=args.ext)

            for layer_name, cl in style_losses.items():
                content_score += cl.loss

            style_score   *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                logprint("run {}:".format(run))
                logprint('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
            return style_score + content_score

        optimizer.step(closure)
        if run[0] % 100 == 0:
            input_img.data.clamp_(0, 1)
            content_name = os.path.split(args.content)[1].split('.')[0] 
            style_name = os.path.split(args.style)[1].split('.')[0]
            out_path = "%s/%s__%s__%s_iter%d.jpg" % (logger.gen_img_path, content_name, style_name, args.net, run[0])
            vutils.save_image(input_img, out_path)

def filter_visualization(net):
    layer_ix = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_ix += 1
            visualize_filter(m, layer_ix, save_dir=logger.gen_img_path, n_filter_plot=16, n_channel_plot=16, prefix=prefix, ext=args.ext)
            print('[%2d] saving visualizations for layer "%s"' % (layer_ix, name))

def analyze_gram(gram, layer_id):
    '''gram is pytorch tensor
    '''
    gram = gram.clone().detach()
    U, E, V = torch.svd(gram)
    fig, ax = plt.subplots()
    ax.plot(E.data.cpu().numpy())
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(layer_id)
    ax.grid(linestyle='dashed')
    save_path = '%s/gram_SVD__%s__layer%s' % (logger.gen_img_path, prefix, layer_id) # prefix is usually a net name
    fig.savefig(save_path + args.ext, bbox_inches='tight')
    plt.close(fig)

######################################################################
parser = argparse.ArgumentParser(description="Gatys NST")
parser.add_argument('--num_steps', type=int, default=1000)
parser.add_argument('--style_weight', type=float, default=1e6, help='default value refers to the PyTorch official tutorial of NST')
parser.add_argument('--content_weight', type=float, default=1)
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--content', type=str)
parser.add_argument('--style', type=str)
parser.add_argument('--net', type=str, default='vgg19', help='network name to choose different CNN models')
parser.add_argument('--fft', action='store_true')
parser.add_argument('--ext', type=str, default='.jpg')
parser.add_argument('--plot_filter', action='store_true')
parser.add_argument('--plot_feature', action='store_true')

# routine args to set up experiment folder
parser.add_argument('--project_name', type=str, default="")
parser.add_argument('--CodeID', type=str, default="")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--screen_print', action="store_true")
parser.add_argument('--resume_ExpID', type=str)
parser.add_argument('--note', type=str, default='', help='experiment note')
args = parser.parse_args()

logger = Logger(args)
logprint = logger.log_printer.logprint
netprint = logger.log_printer.netprint
prefix = args.net + '_fft' if args.fft else args.net + '_nofft' # when saving figures, this will be used as name prefix

if __name__ == "__main__":
    # Additionally, VGG networks are trained on images with each channel
    # normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    # We will use them to normalize the image before sending it into the network.
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std  = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    # set up the model and corresponding layers to compute style/content losses
    model = models.__dict__[args.net]
    if args.net in ['vgg19', 'alexnet']:
        cnn = model(pretrained=True).features.to(device).eval()
        content_layers_default = ['conv_4']
        style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    else:
        raise NotImplementedError
    netprint("Original network arch:", cnn) # print network arch for check

    # visualize filters
    if args.plot_filter:
        filter_visualization(cnn)

    # the main processing func
    run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                args, content_layers_default, style_layers_default, 
                                num_steps=args.num_steps,
                                style_weight=args.style_weight,
                                content_weight=args.content_weight)
