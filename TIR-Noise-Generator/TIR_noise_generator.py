"""
================================================================================
The original code, and therefore this modified version, is licensed under the 
MIT license, with Copyright (c) 2026 Lijing Cai.

This file contains all functions to generate torch tensors of TIR-specific
sensor noise, adapting the model of [ L. Cai, X. Dong, K. Zhou and X. Cao, 
"Exploring Video Denoising in Thermal Infrared Imaging: Physics-Inspired Noise 
Generator, Dataset, and Model," in IEEE Transactions on Image Processing, 
vol. 33, pp. 3839-3854, 2024 ] to an efficient single-image noise-generation.
It handles different channel-numbers by adding the same noise to all channels of
an image.

Depending on the given stds for sampling, the sampled noise is for adding to 
[0,1]-float-images, or [0,255] or any other value-range.


================= EXAMPLE of USE =================
Can be added in the Dataloader or in the training loop to create the noisy 
counterparts from the clean targets:

CASE 1) degrading an image batch:
    noise_params = GetNoiseSamplingParamsLighterNoise()
    N,C,H,W = hq_image_batch.size()
    noisy_batch = SampleNoise(N, C, H, W, noise_params, device=batch_device, mode="train")
    noisy_batch += hq_image_batch # clean tensor remains unchanged

CASE 2) degrading a single image:
    noise_params = GetNoiseSamplingParamsLighterNoise()
    C,H,W = hq_image.size()
    noisy_image = SampleNoise(1, C, H, W, noise_params, device=image_device, mode="train")[0]
    noisy_image += clean_image # clean tensor remains unchanged


================= OVERVIEW of the FUNCTIONS in this script =================

        ===== Main API =====
SampleNoise(N, C, H, W, noise_params, device, mode="train") -> torch.Tensor
    The main function. Combines and returns all sampled noise types as a single
    torch.Tensor of shape (N, C, H, W). The composite noise consists of:
      - Gaussian noise     : spatially i.i.d., std sampled per image
      - Line/stripe noise  : either horizontal or vertical stripes (50/50 in
                             train mode, horizontal only in val/test), std
                             sampled per image
      - Row noise          : per-row offset, std sampled per image
      - Column noise       : per-column offset, std sampled per image
      - Hill noise (bias)  : smooth center-bright bias field simulating the
                             temperature-induced sensor bias; intensity sampled
                             per image. Controlled by patch_size in noise_params
    In train mode, all stds are drawn uniformly from their respective ranges.
    In val/test mode, fixed values are used for reproducibility.

GetNoiseSamplingParamsLighterNoise() -> dict
    Returns a parameter dictionary with the used stripe- and row/col-noise ranges
    (row_col: [3,8]/255, row/col: [0,5]/255), reduced with respect to the original 
    values from Cai et al. to better match visually observed TIR noise levels.
    (stds expressed for [0,255] images, automatically normalized to [0,1] before
      being returned).

        ===== Normalization utilities =====
DivideNoiseSamplingParams255(noise_params) [in-place]
    Divides all std range and value entries in noise_params by 255, converting
    parameters written for [0,255] pixel ranges to [0,1].

DivideNoiseSamplingParams255_flexible(noise_params) [in-place]
    Same as above but tolerates missing keys (safe to call on partial dicts but 
    can silently propagate errors).

divide_values_by_255(d) [in-place]
    Generic version: divides every scalar, list, or tuple value in a dict by
    255. Can be used on arbitrarily structured parameter dicts.

================= POTENTIAL TODOs for enhancing the model =================
TODO: 1) cache the hill-noise ground pattern per image size instead of
         recomputing it for every batch (it is deterministic given H, W)
      2) verify hill-noise intensity is perceptually strong enough (it is barely
         visible in the total composite noise, unlike in Cai et al.'s examples)
      3) quantize the noise to improve realism (not-integer noise values on
         [0,255] images do not exist in real images and may bias the network)
================================================================================
"""


import torch
import torch.nn.functional as F
import random


def SampleNoise(N:int, C:int, H:int, W:int, noise_params: dict, device, mode="train") -> torch.Tensor:
    """
    Generate combined noise (Gaussian (gaussian noise)+ line/col + row + col (stripe noise) + bias field (hill noise) )
    without applying it to an image yet.

    >>>>> IMPORTANT NOTE: the noise_params['patch_size'] does only affect the bias field: it does simulate that the image is only a part of a bigger image, which (the part) was not taken from the center -> not always centered bias, allows to train for tiling. The other noises are translation invariant END IMPORTANT NOTE <<<<<
    NB: each channel (if >1) recieves the exact same noise matrix

    Parameters:
        N (int): batch size
        C (int): channels
        H (int): height
        W (int): width
        noise_params (dict): argument dictionary with noise ranges (same as before)
        device (torch.device): target device
        mode (str): 'train', 'val', or 'test'

    Returns:
        noise_total (torch.Tensor): noise tensor of shape (N, C, H, W)
    """
    # ---- sanity check on parameters ----
    train_dict_keys = ['gaussian_noise_std', 'row_col_noise_std', 'row_noise_std', 'col_noise_std', 'hill_noise_std']
    max_possible_std = 0
    for noise_key in train_dict_keys: # a max in dict.values() would have been more efficient, but so this is robust agains weird keys stuff in the dict
        if mode == "train":
            max_possible_std = max(max_possible_std, max(noise_params[f"{noise_key}_range"]))
        else: #val
            max_possible_std = max(max_possible_std, noise_params[f"{noise_key}_value"])
    if max_possible_std >= 1.:
        print("----------------------------------------------------------------")
        print("--- IMPORTANT WARNING : you are sampling noise with std >= 1 ---")
        print("--- (as img in [0,1]: noise will not leave much of image)    ---")
        print("----------------------------------------------------------------")
    # ---- sample noise parameters ----
    if mode == "train":
        stdn = torch.empty((N, 1, 1, 1), device=device).uniform_(
            noise_params['gaussian_noise_std_range'][0], noise_params['gaussian_noise_std_range'][1]
        ) # -> different std for each batch-element (image), then same for all channels, rows and columns
        stdnline = torch.empty((N, 1, 1, 1), device=device).uniform_(
            noise_params['row_col_noise_std_range'][0], noise_params['row_col_noise_std_range'][1]
        )
        stdnrow_spatial = torch.empty((N, 1, 1, 1), device=device).uniform_(
            noise_params['row_noise_std_range'][0], noise_params['row_noise_std_range'][1]
        )
        stdncol_spatial = torch.empty((N, 1, 1, 1), device=device).uniform_(
            noise_params['col_noise_std_range'][0], noise_params['col_noise_std_range'][1]
        )
        temperature_Coefficient = torch.empty((N, ), device=device).uniform_(
            noise_params['hill_noise_std_range'][0], noise_params['hill_noise_std_range'][1]
        ) # a list of scalars: 1 per batch-element (image) as intensity of hill noise
        patchsize = noise_params['patch_size'] # adapt patch_size

    else:  # validation/test → fixed values
        stdn = torch.full((N, 1, 1, 1), noise_params['gaussian_noise_std_value'], device=device)
        stdnline = torch.full((N, 1, 1, 1), noise_params['row_col_noise_std_value'], device=device)
        stdnrow_spatial = torch.full((N, 1, 1, 1), noise_params['row_noise_std_value'], device=device)
        stdncol_spatial = torch.full((N, 1, 1, 1), noise_params['col_noise_std_value'], device=device)
        temperature_Coefficient = torch.full((N,), noise_params['hill_noise_std_value'], device=device)
        patchsize = None

    # ---- Gaussian noise ---- (with esplanations)
    noise_gaussian = torch.zeros((N, 1, H, W), device=device) # create zero-matrix everywhere where different sampling of the same std
    noise_gaussian = torch.normal(mean=noise_gaussian, std=stdn.expand_as(noise_gaussian)) # std: in each image from batch: same std, with dimention of the zero-matrix made before (expand_as seems kinda magic, it recognizes that stdn has the first dimention same as noise_gaussian and therefore only expands the same stdn-value within an image and vary the std between images according to stdn
    # and then samples independently a value for each component of the inputed std-matrix from a normal distribution
    noise_gaussian = noise_gaussian.repeat(1, C, 1, 1) # duplicate the same noise (already sampled) across all channels

    # ---- Line or row noise (time-varying in video, here just per sample -> proba 0.5 vertical stripes, 0.5 horizontal) ----
    if (mode == "train") and random.randint(0, 1): # 0.5 proba vertical stripes
        line_noise_time = torch.zeros((N, 1, 1, W), device=device)
        line_noise_time = torch.normal(mean=line_noise_time, std=stdnline.expand_as(line_noise_time))
        line_noise_time = line_noise_time.repeat(1, C, H, 1)
    else: # 0.5 proba line-stripes, if test: only line for comparable results
        line_noise_time = torch.zeros((N, 1, H, 1), device=device)
        line_noise_time = torch.normal(mean=line_noise_time, std=stdnline.expand_as(line_noise_time))
        line_noise_time = line_noise_time.repeat(1, C, 1, W)

    # ---- Row noise (horizontal striping)
    row_noise_spatial = torch.zeros((N, 1, H, 1), device=device)
    row_noise_spatial = torch.normal(mean=row_noise_spatial, std=stdnrow_spatial.expand_as(row_noise_spatial))
    row_noise_spatial = row_noise_spatial.repeat(1, C, 1, W)

    # ---- Column noise (vertical banding) ----
    col_noise_spatial = torch.zeros((N, 1, 1, W), device=device)
    col_noise_spatial = torch.normal(mean=col_noise_spatial, std=stdncol_spatial.expand_as(col_noise_spatial))
    col_noise_spatial = col_noise_spatial.repeat(1, C, H, 1)

    # ---- Bias field ----
    bias_field = SampleHillNoiseField(N, C, H, W, temperature_Coefficient, patchsize=patchsize, device=device).to(device) # normally already on device, but tbs...

    # ---- total noise ----
    noise_total = noise_gaussian + line_noise_time + row_noise_spatial + col_noise_spatial + bias_field # all tensors already on device -> torch keeps the device for result: even if it is a fresh tensor, it lives on device

    return noise_total



def GetNoiseSamplingParamsLighterNoise() -> dict:
    """ Return a dictionnary with the different sampling ranges for the noise-generator"""
    noise_params = {}

    # sampling parameters for random noise generation
    # (already floats bcse then: changes to float in place -> would type conflict else)
    noise_params['gaussian_noise_std_range'] = [5., 30.] # gaussian-noise-std-range for uniform sampling
    noise_params['row_col_noise_std_range'] = [3., 8.] # line/col-noise-std-range for uniform sampling
    noise_params['row_noise_std_range'] = [0., 5.] # row-noise-std-range for uniform sampling
    noise_params['col_noise_std_range'] = [0., 5.] # col-noise-std-range for uniform sampling
    noise_params['hill_noise_std_range'] = [5., 35.] # hill-noise-std-range for uniform sampling

    # fixed validation/test parameters -> to have comparable results when evaluating
    noise_params['gaussian_noise_std_value'] = 15.
    noise_params['row_col_noise_std_value'] = 5.
    noise_params['row_noise_std_value'] = 3.
    noise_params['col_noise_std_value'] = 3.
    noise_params['hill_noise_std_value'] = 20.

    # Normalize noise between [0, 1]
    DivideNoiseSamplingParams255(noise_params) # divide all params by 255 (in-place)

    # /!\/!\/!\ the patch_size does only affect the bias field: it simulates that the image is only a part of a bigger image, which (the part) was not taken from the center -> not always centered bias, allows to train for tiling. The other noises are translation invariant. None makes the bias field being centered and in general behaves like the image was the full information returned by the infrared sensor /!\/!\/!\
    noise_params['patch_size'] = None # change to 256 for example

    return noise_params




#===============================================================================
# Helper-Functions used in SampleNoise() and the param-dictionnary-returning functions


def GenNormalizedHillNoiseField(size: "indexable of size 2", device="cpu") -> torch.Tensor: # type: ignore (to suppress warning)
    """
    Creates a synthetic NORMALIZED bias field map to simulate hill-noise 
    (smooth background pattern). 
    The randomness of noise-intensity happens in SampleHillNoiseField.

    Vectorized + GPU-ready version of the original numpy implementation.

    Args:
        size (tuple): (H, W) output size
        device (str or torch.device): 'cpu' or 'cuda'

    Returns:
        torch.Tensor: (H, W) bias field in float32 on chosen device
    """
    H, W = size
    size_center = min(size) // 2

    # ---- Define coil positions (inner + outer) ----
    num_points = 4
    cx = torch.full((num_points,), size_center, device=device, dtype=torch.float32)
    cy = torch.full((num_points,), size_center, device=device, dtype=torch.float32)
    coils = torch.stack([cy, cx], dim=1)  # (4, 2)

    cy_out = torch.tensor([size_center, -3*size_center, 5*size_center, size_center],
                          device=device, dtype=torch.float32)
    cx_out = torch.tensor([-3*size_center, size_center, size_center, 5*size_center],
                          device=device, dtype=torch.float32)
    coils_out = torch.stack([cy_out, cx_out], dim=1)  # (4, 2)

    # ---- Build coordinate grid ----
    yy, xx = torch.meshgrid(
        torch.arange(min(size), device=device, dtype=torch.float32),
        torch.arange(min(size), device=device, dtype=torch.float32),
        indexing="ij"
    )
    coords = torch.stack([yy, xx], dim=-1)  # (H, W, 2)

    # ---- Distance to nearest coil (inner & outer) ----
    dist1 = torch.cdist(coords.reshape(-1, 2), coils).min(dim=1).values
    dist2 = torch.cdist(coords.reshape(-1, 2), coils_out).min(dim=1).values

    dist1 = dist1.clamp_min(10.0)  # avoid too small values
    B = -torch.log(dist2 / dist1)

    # ---- Normalize & enhance contrast ---- => weird: didn't do in paper!!!
    B = B.view(min(size), min(size))
    B_norm = (B - B.min()) / (B.max() - B.min() + 1e-8) # all values in [0,1]
    B_norm = B_norm.pow(4)

    # ---- Resize to requested size (H, W) ----
    B_resized = F.interpolate(
        B_norm.unsqueeze(0).unsqueeze(0),  # (1,1,H,W)
        size=(H, W),
        mode="bilinear",
        align_corners=False
    ).squeeze(0).squeeze(0)

    return B_resized  # (H, W), float32


def SampleHillNoiseField(N:int, C:int, H:int, W:int, intensity: torch.Tensor, patchsize: "int or None" = None, device='cpu') -> torch.Tensor: # type: ignore
    """
    Samples batch of bias fields with optional random patch cropping:
    randomly scales the normalized hill-noise-bias-field (outputed by GenNormalizedHillNoiseField)
    and randomly crops a patch of patchsize^2 somewhere in the image

    Args:
        N: batch size
        C: channels (will repeat bias across channels)
        H, W: output size
        patchsize: if given, crop a random (patchsize x patchsize) region
        device: 'cpu' or 'cuda'
    """
    bias_field = torch.zeros((N, C, H, W), device=device)

    # Generate full-size field (on device)
    B_base = GenNormalizedHillNoiseField((H, W), device=device)  # normalized hill-noise-field, (H, W), computed once in function -> once/batch
    # TODO: here: 1 Bias field generated for each batch (GenNormalizedHillNoiseField). BUT: the random component : happens only here -> GenNormalizedHillNoiseField(...) outputs a constant value, as long as image size: constant
    #  -> include an option to say if all images have same size or if have to compute this every time. 
    # Else: can store somewhere a list of seen sizes and store the corresponding map together with it -> adds a lookup per image, but computing-efficient

    for i in range(N):
        B_i = B_base.clone()  # <---- make a copy for this sample
        if patchsize is not None and patchsize < min(H, W):
            # Pick random bottom-left corner
            nh = random.randint(0, H - patchsize)
            nw = random.randint(0, W - patchsize)
            B_i = B_i[nh:nh+patchsize, nw:nw+patchsize]

            # Resize patch back to (H, W) so dimensions match
            B_i = torch.nn.functional.interpolate(
                B_i.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).squeeze(0)

        # Apply intensity scaling -> denormalize and recenter around 0
        B_i = B_i * intensity[i] - intensity[i] / 2

        # Store into batch, repeat for channels
        bias_field[i] = B_i.unsqueeze(0).repeat(C, 1, 1)

    return bias_field


def DivideNoiseSamplingParams255(noise_params:dict):
    """Normalizes the noise_params by dividing them by 255.
    Modification are made in place, as Python passes a reference to noise_params
    when calling this function, and does not make a copy (according to ChatGPT)"""
 
    noise_params['gaussian_noise_std_range'][0] /= 255. # here: force convert to float in place -> must be float before
    noise_params['gaussian_noise_std_range'][1] /= 255.
    noise_params['row_col_noise_std_range'][0] /= 255.
    noise_params['row_col_noise_std_range'][1] /= 255.
    noise_params['row_noise_std_range'][0] /= 255.
    noise_params['row_noise_std_range'][1] /= 255.
    noise_params['col_noise_std_range'][0] /= 255.
    noise_params['col_noise_std_range'][1] /= 255.
    noise_params['hill_noise_std_range'][0] /= 255.
    noise_params['hill_noise_std_range'][1] /= 255.

    noise_params['gaussian_noise_std_value'] /= 255.
    noise_params['row_col_noise_std_value'] /= 255.
    noise_params['row_noise_std_value'] /= 255.
    noise_params['col_noise_std_value'] /= 255.
    noise_params['hill_noise_std_value'] /= 255.

def DivideNoiseSamplingParams255_flexible(noise_params:dict):
    """Does the same as DivideNoiseSamplingParams255 but allows some keys to not be there"""

    noise_key_prefix_list = ['gaussian_noise_std', 'row_col_noise_std', 'row_noise_std', 'col_noise_std', 'hill_noise_std']
    existing_keys = noise_params.keys()

    for noise_key_prefix in noise_key_prefix_list:

        key_range = f"{noise_key_prefix}_range"
        if key_range in existing_keys:
            noise_params[key_range][0] /= 255.
            noise_params[key_range][1] /= 255.

        key_value = f"{noise_key_prefix}_value"
        if key_value in existing_keys:
            noise_params[key_value] /= 255.

def divide_values_by_255(d: dict):
    """
    Divides each value of the dictionary by 255.
    - tuple  -> divides each element
    - list   -> divides each element
    - scalar -> divides value
    Other types are ignored.
    """

    for key, value in d.items():

        # Case 1: tuple → elementwise
        if isinstance(value, tuple):
            d[key] = tuple(v / 255. for v in value)

        # Case 2: list → elementwise
        elif isinstance(value, list):
            d[key] = [v / 255. for v in value]

        # Case 3: scalar
        elif isinstance(value, (int, float)):
            d[key] = value / 255.