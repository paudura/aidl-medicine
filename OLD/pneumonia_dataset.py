import os 

import os
import time

import skimage 
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import PIL
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset as torchDataset
import torchvision as tv
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import shutil
import pydicom
import warnings
import data_input as di
import auxiliar as aux



# define the data generator class
class PneumoniaDataset(torchDataset):
    """
        Pneumonia dataset that contains radiograph lung images as .dcm. 
        Each patient has one image named patientId.dcm.
    """

    def __init__(self, root, subset, pIds, predict, boxes, rescale_factor=1, transform=None, rotation_angle=0, warping=False):
        """
        :param root: it has to be a path to the folder that contains the dataset folders
        :param subset: 'train' or 'test'
        :param pIds: list of patient IDs
        :param predict: boolean, if true returns images and target labels, otherwise returns only images
        :param boxes: a {patientId : list of boxes} dictionary (ex: {'pId': [[x1, y1, w1, h1], [x2, y2, w2, h2]]}
        :param rescale_factor: image rescale factor in network (image shape is supposed to be square)
        :param transform: transformation applied to the images and their target masks
        :param rotation_angle: float, defines range of random rotation angles for augmentation (-rotation_angle, +rotation_angle)
        :param warping: boolean, whether applying augmentation warping to image
        """
        
        # initialize variables
        self.root = os.path.expanduser(root)
        self.subset = subset
        if self.subset not in ['train', 'test']:
            raise RuntimeError('Invalid subset ' + self.subset + ', it must be one of: \'train\' or \'test\'')
        self.pIds = pIds
        self.predict = predict
        self.boxes = boxes
        self.rescale_factor = rescale_factor
        self.transform = transform
        self.rotation_angle = rotation_angle
        self.warping = warping

        self.data_path = self.root + 'stage_2_'+self.subset+'_images/'
        
    def __getitem__(self, index):
        # get the corresponding pId
        pId = self.pIds[index]
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.data_path, pId+'.dcm')).pixel_array
        # check if image is square
        if (img.shape[0]!=img.shape[1]):
            raise RuntimeError('Image shape {} should be square.'.format(img.shape))
        original_image_shape = img.shape[0]
        # calculate network image shape
        image_shape = original_image_shape / self.rescale_factor
        # check if image_shape is an integer
        if (image_shape != int(image_shape)):
            raise RuntimeError('Network image shape should be an integer.'.format(image_shape))
        image_shape = int(image_shape)
        # resize image 
        # IMPORTANT: skimage resize function rescales the output from 0 to 1, and pytorch doesn't like this!
        # One solution would be using torchvision rescale function (but need to differentiate img and target transforms)
        # Here I use skimage resize and then rescale the output again from 0 to 255
        img = resize(img, (image_shape, image_shape), mode='reflect')
        # recale image from 0 to 255
        img = aux.imgMinMaxScaler(img, (0,255))
        # image warping augmentation
        if self.warping:
            img = aux.elastic_transform(img, image_shape*2., image_shape*0.1)
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        # apply rotation augmentation
        if self.rotation_angle > 0:
            angle = self.rotation_angle * (2 * np.random.random_sample() - 1) # generate random angle 
            img = tv.transforms.functional.to_pil_image(img)
            img = tv.transforms.functional.rotate(img, angle, resample=PIL.Image.BILINEAR)
                                            
        # apply transforms to image
        if self.transform is not None:
            img = self.transform(img)
        
        if not self.predict:
            # create target mask
            target = np.zeros((image_shape, image_shape))
            # if patient ID has associated target boxes (=if image contains pneumonia)
            if pId in self.boxes:
                # loop through boxes
                for box in self.boxes[pId]:
                    # extract box coordinates 
                    x, y, w, h = box
                    # rescale box coordinates
                    x = int(round(x/self.rescale_factor))
                    y = int(round(y/self.rescale_factor))
                    w = int(round(w/self.rescale_factor))
                    h = int(round(h/self.rescale_factor))
                    # create a mask of 1s (255 is used because pytorch will rescale to 0-1) inside the box
                    target[y:y+h, x:x+w] = 255 #
                    target[target>255] = 255 # correct in case of overlapping boxes (shouldn't happen)
            # add trailing channel dimension
            target = np.expand_dims(target, -1)   
            target = target.astype('uint8')
            # apply rotation augmentation
            if self.rotation_angle>0:
                target = tv.transforms.functional.to_pil_image(target)
                target = tv.transforms.functional.rotate(target, angle, resample=PIL.Image.BILINEAR)
            # apply transforms to target
            if self.transform is not None:
                target = self.transform(target)
            return img, target, pId
        else: 
            return img, pId

    def __len__(self):
        return len(self.pIds)