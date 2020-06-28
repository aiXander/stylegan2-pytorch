from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import utils
import torch.utils.cpp_extension
import random
import os

import torchvision.transforms.functional as TF
import kornia
import numpy as np

class Squash_Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_perc  = torch.cuda.FloatTensor(1).fill_(0.25)

    def forward(self, x, params, indicies):
        bs, n_channels, w, h = x.shape
        self.probs = torch.cuda.FloatTensor(bs, n_channels).uniform_().unsqueeze(2).unsqueeze(3)
        self.probs = self.probs.repeat(1, 1, w, h)
        self.zeros = torch.cuda.FloatTensor(bs, n_channels, w, h).fill_(0)

        x = torch.where(self.probs > self.feature_perc, x, self.zeros)
        
        return x

class Translate(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_shift  = 0.
        self.y_shift  = 0.5

    def forward(self, x, params, indicies):
        bs, n_channels, w, h = x.shape
        shifts = (int(self.x_shift * w), int(self.y_shift*h))
        return torch.roll(x, shifts, dims=[2,3])

class Rotate(nn.Module):
    def __init__(self):
        super().__init__()
        self.angle  = torch.cuda.FloatTensor(1).fill_(0)
        self.scale  = torch.cuda.FloatTensor(1).fill_(1)
        self.center = torch.cuda.FloatTensor(1, 2).fill_(0)

    def update(self, angle, scale=1):
        #Overwrite previous pass
        self.angle.fill_(angle)
        self.scale.fill_(scale)
    
    def forward(self, x, params, indicies):
        # define the rotation center
        self.center[..., 0] = x.shape[3] / 2
        self.center[..., 1] = x.shape[2] / 2

        M = kornia.get_rotation_matrix2d(self.center, self.angle, self.scale)
        return kornia.warp_affine(x, M, dsize = (x.shape[2],x.shape[3]), flags = 'bilinear', padding_mode = 'reflection')


class ManipulationLayer(nn.Module):
    def __init__(self, layerID):
        super().__init__()
        self.layerID = layerID
        
        # layers
        self.rotate = Rotate()
        self.translate = Translate()
        self.random_squash = Squash_Features()
        
        self.layer_options = {
            "rotate": self.rotate,
            "translate": self.translate,
            "random_squash": self.random_squash
        }

    def save_activations(self, input, index):
        path = 'activations/'+str(self.layerID) +'/'+str(index) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        
        x_array = list(torch.split(input,1,1))
        for i, activation in enumerate(x_array):
            utils.save_image(
                torch.squeeze(activation),
                path+str(i).zfill(3)+'.png',
                nrow=1,
                normalize=True,
                range=(-1, 1))

    def forward(self, input, tranforms_dict_list):
        out = input
        for transform_dict in tranforms_dict_list:
            if transform_dict['layerID'] == -1:
                self.save_activations(input, transform_dict['index'])
            if transform_dict['layerID'] == self.layerID:
                out = self.layer_options[transform_dict['transformID']](out, transform_dict['params'], transform_dict['indicies'])
        return out


'''
torch.ops.load_library("transforms/erode/build/liberode.so")
torch.ops.load_library("transforms/dilate/build/libdilate.so")
torch.ops.load_library("transforms/scale/build/libscale.so")
torch.ops.load_library("transforms/rotate/build/librotate.so")
torch.ops.load_library("transforms/translate/build/libtranslate.so")

class Erode(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isinstance(params[0], int) or params[0] < 0):
            print('Erosion parameter must be a positive integer')
            # raise ValueError
        
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = torch.ops.my_ops.erode(d_,params[0])
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class Dilate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isinstance(params[0], int) or params[0] < 0):
            print('Dilation parameter must be a positive integer')
            # raise ValueError
        
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = torch.ops.my_ops.dilate(d_,params[0])
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class Translate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isinstance(params[0], float) or not isinstance(params[1], float)
         or params[0] < -1 or params[0] > 1 or params[1] < -1 or params[1] > 1):
            print('Translation must have two parameters, which should be floats between -1 and 1.')
            # raise ValueError
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = torch.ops.my_ops.translate(d_,params[0], params[1])
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class Scale(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isinstance(params[0], float)):
            print('Scale parameter should be a float.')
            # raise ValueError
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = torch.ops.my_ops.scale(d_,params[0])
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class Rotate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isinstance(params[0], float) or params[0] < 0 or params[0] > 360):
            print('Rotation parameter should be a float between 0 and 360 degrees.')
            # raise ValueError
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = torch.ops.my_ops.rotate(d_,params[0])
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class FlipHorizontal(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = d_.flip([1])
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class FlipVertical(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = d_.flip([0])
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class Invert(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                ones = torch.ones(d_.size(), dtype=d_.dtype, layout=d_.layout, device=d_.device)
                tf = ones - d_
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class BinaryThreshold(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isinstance(params[0], float) or params[0] < -1 or params[0] > 1):
            print('Binary threshold parameter should be a float between -1 and 1.')
            # raise ValueError

        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                t = Variable(torch.Tensor([params[0]]))
                t = t.to(d_.device)
                tf = (d_ > t).float() * 1
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class ScalarMultiply(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isinstance(params[0], float)):
            print('Scalar multiply parameter should be a float')
            # raise ValueError

        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = d_ * params[0]
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class Ablate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = d_ * 0
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class ManipulationLayer(nn.Module):
    def __init__(self, layerID):
        super().__init__()
        self.layerID = layerID
        
        # layers
        self.erode = Erode()
        self.dilate = Dilate()
        self.translate = Translate()
        self.scale = Scale()
        self.rotate = Rotate()
        self.flip_h = FlipHorizontal()
        self.flip_v = FlipVertical()
        self.invert = Invert()
        self.binary_thresh = BinaryThreshold()
        self.scalar_multiply = ScalarMultiply()
        self.ablate = Ablate()
        
        self.layer_options = {
            "erode" : self.erode,
            "dilate": self.dilate,
            "translate": self.translate,
            "scale": self.scale,
            "rotate": self.rotate,
            "flip-h": self.flip_h,
            "flip-v": self.flip_v,
            "invert": self.invert,
            "binary-thresh": self.binary_thresh,
            "scalar-multiply": self.scalar_multiply,
            "ablate": self.ablate
        }

    def save_activations(self, input, index):
        path = 'activations/'+str(self.layerID) +'/'+str(index) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        
        x_array = list(torch.split(input,1,1))
        for i, activation in enumerate(x_array):
            utils.save_image(
                torch.squeeze(activation),
                path+str(i).zfill(3)+'.png',
                nrow=1,
                normalize=True,
                range=(-1, 1))

    def forward(self, input, tranforms_dict_list):
        out = input
        for transform_dict in tranforms_dict_list:
            if transform_dict['layerID'] == -1:
                self.save_activations(input, transform_dict['index'])
            if transform_dict['layerID'] == self.layerID:
                out = self.layer_options[transform_dict['transformID']](out, transform_dict['params'], transform_dict['indicies'])
        return out

'''