'''
Copyright (c) 2022 Zhendong Wang
'''

"""
================================================================================
This script is modified from the original repo to reroute the code towards local
data instead of the SIDD dataset the code is tailored for.

Modifications include :
- a modification of the gt- and noisy-folder inside to look into, inside the 
training- and validation-folder passed as arguments
- a center croping in the DataLoaderVal class to force a dimension match with 
the SIDD validation resolution of 256x256
================================================================================
"""

import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms.functional as TF
from natsort import natsorted
from glob import glob

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../TIR-Noise-Generator')))
from TIR_noise_generator import SampleNoise, GetNoiseSamplingParamsLighterNoise
# from train_8bit_denoising import scale_tensor_global, scale_tensor # if need to resize


augment   = Augment_RGB_torch() # check if dimensions also match dim 1 images, but should
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])
    
##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir_HQ = "path/to/clean/targets", rgb_dir_LQ = None, img_options=None, target_transform=None, n2n:bool = False):
        """
        Docstring for __init__
        
        :param self: obvious
        :param rgb_dir_HQ: directory to clean target images
        :param rgb_dir_LQ: directory to noisy images (paired by name with the HQ ones)
                            if None: the dataset applies a degradation on the 
                            clean image to synthesize the LQ one
        :param img_options: passed from before, was in the original code
        :param target_transform: same
        :param n2n: if training according to the Noise2Noise framework (then, the
                    target image undergoes the same degradation as the input, 
                    but sampled independently)
        """
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        self.need_degradation = (rgb_dir_LQ is None)
        self.n2n = n2n

        clean_files = sorted(os.listdir(rgb_dir_HQ))
        self.clean_filenames = [os.path.join(rgb_dir_HQ, x) for x in clean_files if is_png_file(x)]
        
        if not self.need_degradation:
            noisy_files = sorted(os.listdir(rgb_dir_LQ))
            self.noisy_filenames = [os.path.join(rgb_dir_LQ, x) for x in noisy_files if is_png_file(x)] # type: ignore
        else:
            self.noise_params = GetNoiseSamplingParamsLighterNoise()
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        clean = clean.permute(2,0,1) # check if that is ok for grayscale
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]

        # --- Crop Target (Input: later) ---          -> crop again patch-size, also random
        ps = self.img_options['patch_size'] # type: ignore
        C, H, W = clean.size()
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]

        # --- Add degradation if needed ---
        c, h, w = clean.size() # redefine new
        if not self.need_degradation:
            noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
            noisy = noisy.permute(2,0,1)
            noisy = noisy[:, r:r + ps, c:c + ps] # also crop the noisy input
            noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        else:
            noisy = SampleNoise(1, c, h, w, self.noise_params, device='cpu', mode="train")[0]
            noisy += clean
            noisy_filename = clean_filename

        if self.n2n:
            clean += SampleNoise(1, c, h, w, self.noise_params, device='cpu', mode="train")[0]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        

        return clean, noisy, clean_filename, noisy_filename


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir_HQ = "/home/go69niz/Masterarbeit-Datasets/UNet_train_data/val_clean", rgb_dir_LQ = "/home/go69niz/Masterarbeit-Datasets/UNet_train_data/val_noisy_G5,30-LC3,8-L0,5-C0,5-HN5,35", target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform
        
        clean_files = sorted(os.listdir(rgb_dir_HQ))
        noisy_files = sorted(os.listdir(rgb_dir_LQ))

        self.clean_filenames = [os.path.join(rgb_dir_HQ, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir_LQ, x) for x in noisy_files if is_png_file(x)]

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        
        # --- load images to torch ---
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        # --- center crop to 256x256 to mimic the validation data of SIDD ---
        clean = center_crop(clean, 256, 256)
        noisy = center_crop(noisy, 256, 256)
        # --- order to rgb ---
        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        # filenames
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        return clean, noisy, clean_filename, noisy_filename

def center_crop(img, crop_h=256, crop_w=256):
    h, w, _ = img.shape
    top  = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return img[top:top + crop_h, left:left + crop_w, :]

##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)
        return inp, filename


def get_training_data(rgb_dir_HQ, rgb_dir_LQ, img_options, n2n:bool=False):
    assert os.path.exists(rgb_dir_HQ)
    return DataLoaderTrain(rgb_dir_HQ, rgb_dir_LQ, img_options, target_transform=None, n2n=n2n)


def get_validation_data(rgb_dir_HQ, rgb_dir_LQ):
    assert os.path.exists(rgb_dir_HQ)
    assert os.path.exists(rgb_dir_LQ)
    return DataLoaderVal(rgb_dir_HQ, rgb_dir_LQ, target_transform=None)

def get_test_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)