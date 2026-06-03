"""
================================================================================
This file contains everything to load and store image-batches:
    - the dataset-class SingleTIRDataset(Dataset) for loading a batch of images 
from a folder with the associated names. Returns (image, name).
    - the dataset-class PairedTIRDataset(Dataset) for loading paired HQ/LQ image 
batches from two folders. Both folders must contain identically named files. 
Returns (hq_image, lq_image, name).
    - a dataloader wrapper function GetDataLoader(...) creating a DataLoader 
instance from a given dataset, with configurable batch size, shuffling, and 
optional custom collation.
    - a custom normalization transformation (class NormalizeTensor) that 
normalizes a float tensor by a given max_value (e.g. 255 or 16383 for 14-bit), 
instead of relying on dtype-based rescaling. To be passed at 
Dataset-initialization.
    - a fixed-position cropping transform (class RandomCrop) that crops to a 
fixed H×W at a uniformly sampled random position. To be passed at 
Dataset-initialization. The fixed H×W ensures same tensor shapes needed for 
later stacking into a batch in the Dataloader class.
    - a batch cropping function random_crop_batch(...) designed to be called 
during training after fetching a batch from the DataLoader. Crops all images 
in a (N,C,H,W) batch to the same H×W, either at a shared random center or at 
per-image independent centers (different_centers=True). To be passed at 
Dataset-initialization.
    - a fixed-position cropping transform (class FlexibleCrop) that crops to a 
fixed H×W at a configurable position: "center", "topleft", "topright", 
"bottomleft", or "bottomright". To be passed at Dataset-initialization.
    - a helper function uniform_sample_size(...) for uniformly sampling a crop 
size along one dimension in [min_fraction*dim, dim].
    - a function NameAndSaveImage(...) that saves an image-batch tensor as 
individual PNG images. Supports bit depths 8 or 16. Images can be named via an 
explicit name list or auto-named with an incrementing index. Tensors are 
normalized to [0,1] before saving.
================================================================================
"""


import os
import torch
    # Dataset
from torch.utils.data.dataset import Dataset
from torchvision.io import decode_image, read_file
    # Dataloader
from torch.utils.data import DataLoader
    #for saving images
from PIL import Image
    # for crop transformation
from statistics import NormalDist
import random
from torchvision.transforms.v2 import functional as F


#=========================== DATA-AUGM. - TRANSFORMS ===========================
class NormalizeTensor:
    """ to normalize an image stored with pixel-values stored as uint8 or uint16.
     
    Args:
        max_value (int): is the maximum value with which the pixel-values are 
                        normalized. 
    /!\\ as the function is called very often, we do not check if max_value >= max(tensor)!

    NB:
        We could use:
        from torchvision.transforms.v2 import functional as F
        image = decode_image(img_path)  # uint8 [0..255] or uint16 [0..65535]
        image = F.to_dtype(image, dtype=torch.float32, scale=True) # in [0,1]
        But there, the normalization is done via the datatype -> for tir-raw 
        images, stored in uint16, but effectively using only 14 bit, it is useless
        -> custom tranform
     """
    def __init__(self, max_value=255):
        self.max_value = float(max_value)

    def __call__(self, tensor: torch.Tensor):
        # ATTENTION: NO CHECK THAT max_value >= max(tensor) -> be sure when use
        tensor = tensor.to(torch.float32)
        
        # --- sanity check ---
        tmax = tensor.max().item()
        if tmax > self.max_value:
            print(f"Input-image has max value {tmax} while given max for normalization is {self.max_value}.")
            input("press Enter to continue")
        
        return tensor / self.max_value


class RandomCrop:
    """
    Random cropping transform for data augmentation. 
    (Given as arg at Dataset-initialization.)

    Args:
        crop_height (int, optional): Fixed crop height.
        crop_width (int, optional): Fixed crop width.
    """

    def __init__(self, crop_height=512, crop_width=640):
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (torch.Tensor): Image tensor [C,H,W] or [H,W].

        Returns:
            Cropped tensor.
        """
        if tensor.ndim == 2:
            H, W = tensor.shape
        elif tensor.ndim == 3:
            C, H, W = tensor.shape
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

        # --- Clamp to image size ---
        crop_h = min(self.crop_height, H)
        crop_w = min(self.crop_width, W)
        # --- Sample crop position ---
        top = random.randint(0, H - crop_h) if H > crop_h else 0
        left = random.randint(0, W - crop_w) if W > crop_w else 0
        # --- Crop ---
        return F.crop(tensor, top, left, crop_h, crop_w)


class FlexibleCrop:
    """
    Cropping transform for data augmentation with configurable position.

    Args:
        crop_height (int): Desired crop height.
        crop_width (int): Desired crop width.
        position (str): Crop position. One of:
            "center", "topleft", "topright", "bottomleft", "bottomright"
    """

    def __init__(self, crop_height=512, crop_width=640, position="center"):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.position = position.lower()

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (torch.Tensor): Image tensor [C,H,W] or [H,W].

        Returns:
            torch.Tensor: Cropped tensor.
        """
        if tensor.ndim == 2:
            H, W = tensor.shape
        elif tensor.ndim == 3:
            C, H, W = tensor.shape
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

        # --- Clamp to image size ---
        crop_h = min(self.crop_height, H)
        crop_w = min(self.crop_width, W)

        # --- Determine crop position ---
        if self.position == "center":
            top = (H - crop_h) // 2
            left = (W - crop_w) // 2
        elif self.position == "topleft":
            top, left = 0, 0
        elif self.position == "topright":
            top, left = 0, W - crop_w
        elif self.position == "bottomleft":
            top, left = H - crop_h, 0
        elif self.position == "bottomright":
            top, left = H - crop_h, W - crop_w
        else:
            raise ValueError(f"Invalid position '{self.position}'")

        # --- Crop ---
        return F.crop(tensor, top, left, crop_h, crop_w)


def uniform_sample_size(dim_size, fixed_size=None, min_fraction=0.25):
    """
    Uniformly sample a size for cropping, between min_fraction*dim_size and dim_size.

    Args:
        dim_size (int): Original dimension.
        fixed_size (int or None): If given, return this size.
        min_fraction (float): Minimum crop size as fraction of dim_size (default: 0.25).

    Returns:
        int: Chosen crop size.
    """
    if fixed_size is not None:
        return fixed_size

    min_size = int(dim_size * min_fraction)
    sampled = random.randint(min_size, dim_size)
    return sampled


def random_crop_batch(batch: torch.Tensor, crop_height=None, crop_width=None, different_centers=False):
    """
    Randomly crop a batch of images.      -> Use in training: 
    cropped_batch = random_crop_batch(batch, crop_height=128, crop_width=128)

    Args:
        batch (torch.Tensor): Input of shape (N, C, H, W).
        crop_height (int, optional): If given, fixed crop height. Otherwise sampled.
        crop_width  (int, optional): If given, fixed crop width. Otherwise sampled.
        different_centers (bool): If True, each image gets its own crop center.
                                  If False, same center for all in batch -> better

    Returns:
        torch.Tensor: Cropped batch (N, C, crop_height, crop_width).
    """
    N, C, H, W = batch.shape

    crop_h = uniform_sample_size(H, fixed_size=crop_height, min_fraction=0.25)
    crop_w = uniform_sample_size(W, fixed_size=crop_width, min_fraction=0.25)
    #crop_h = gaussian_sample_size(H, fixed_size=crop_height, min_fraction=0.25, target_prob=0.99)
    #crop_w = gaussian_sample_size(W, fixed_size=crop_width, min_fraction=0.25, target_prob=0.99)

    if not different_centers:
        # same crop center for all
        top = random.randint(0, H - crop_h) if H > crop_h else 0
        left = random.randint(0, W - crop_w) if W > crop_w else 0
        return batch[:, :, top:top + crop_h, left:left + crop_w]
    else:
        # per-sample crop centers
        out = []
        for i in range(N):
            top = random.randint(0, H - crop_h) if H > crop_h else 0
            left = random.randint(0, W - crop_w) if W > crop_w else 0
            out.append(batch[i:i+1, :, top:top + crop_h, left:left + crop_w])
        return torch.cat(out, dim=0)


#=============================== DATASET CLASSES ===============================
class SingleTIRDataset(Dataset):
    """Creates a Dataset for single-image-loading: 
    __getitem__(...) returns only an image and the corresponding image name"""
    def __init__(self, img_dir, input_transform=None):
        self.img_dir = img_dir
        # Sort for deterministic order
        self.file_list = sorted(
            [name for name in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, name)) 
                                                            and os.path.splitext(name)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}]
        )
        self.input_transform = input_transform #transforms to be applied to the data when __getitem__

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, image_name)
        # --- Load image with PIL (supports 8-bit and 16-bit) ---
        pil_img = Image.open(img_path)
        # --- Convert to tensor ---
        # --- Convert to tv_tensors.Image (internally stores tensor CxHxW) ---
        image = F.to_image(pil_img)
        # --- Convert to float32 in [0,1] ---
        # scale=True rescales from dtype max (e.g. 65535 → 1.0 for 16-bit) -> I do not want that
        image = F.to_dtype(image, torch.float32, scale=False)
        # --- Apply transforms if provided ---
        if self.input_transform:
            image = self.input_transform(image)

        return image, image_name


class PairedTIRDataset(Dataset):
    """
    Creates a Dataset for paired-image-loading: 
    __getitem__(...) returns two images and the corresponding image name, 
    following the order: (hq, lq, name) 
    Note: the input_transform is applied to both images
    """
    def __init__(self, hq_img_dir, lq_img_dir, input_transform=None):
        self.hq_img_dir = hq_img_dir
        self.lq_img_dir = lq_img_dir
        # Sort for deterministic order
        self.hq_file_list = sorted(
            [name for name in os.listdir(self.hq_img_dir) if os.path.isfile(os.path.join(self.hq_img_dir, name)) 
                                                            and os.path.splitext(name)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}]
        )
        self.lq_file_list = sorted(
            [name for name in os.listdir(self.lq_img_dir) if os.path.isfile(os.path.join(self.lq_img_dir, name)) 
                                                            and os.path.splitext(name)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}]
        )
        #self.hq_file_list.sort() #here, sorting is important to compare. Afterwards, only 1 name list is needed
        #self.lq_file_list.sort()
        assert self.hq_file_list == self.lq_file_list , "The hq and lq folders must contain the same image names to create pairs"
        self.input_transform = input_transform #transforms to be applied to the data when __getitem__

    def __len__(self):
        return len(self.hq_file_list) # both have same length, once passed the assert

    
    def __getitem__(self, idx):
        image_name = self.hq_file_list[idx] # same in lq_ and hq_file_list
        hq_img_path = os.path.join(self.hq_img_dir, image_name)
        lq_img_path = os.path.join(self.lq_img_dir, image_name)
        # --- Load image with PIL (supports 8-bit and 16-bit) ---
        hq_pil_img = Image.open(hq_img_path)
        lq_pil_img = Image.open(lq_img_path)
        # --- Convert to tensor ---
        # --- Convert to tv_tensors.Image (internally stores tensor CxHxW) ---
        hq_image = F.to_image(hq_pil_img)
        lq_image = F.to_image(lq_pil_img)
        # --- Convert to float32 in [0,1] ---
        # scale=True rescales from dtype max (e.g. 65535 → 1.0 for 16-bit) -> do not want that
        hq_image = F.to_dtype(hq_image, torch.float32, scale=False)
        lq_image = F.to_dtype(lq_image, torch.float32, scale=False)
        # --- Apply transforms if provided ---
        if self.input_transform:
            hq_image = self.input_transform(hq_image)
            lq_image = self.input_transform(lq_image)
        return hq_image, lq_image, image_name




#============================= DATALOADER WRAPPER ==============================
def GetDataLoader(dataset: torch.utils.data.dataset.Dataset, batch_size: int = 1, shuffle: bool = True, drop_last:bool=False, num_workers=0, pin_memory=False, collate_fn=None) -> torch.utils.data.DataLoader : # type: ignore to suppress warning
    """wrapper function for creating an instance of the torch.utils.dataDataLoader class
    
    Args:
        dataset (torch.utils.data.dataset.Dataset): dataset from which to load the data
        batch_size (int, optional): how many samples per batch to load (default: 1)
        shuffle (bool, optional): set to True to have the data reshuffled 
            at every epoch (default: False).
        drop_last (bool, optional): set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. 
            If False and the size of dataset is not divisible by the batch size,
            then the last batch will be smaller. (default: False)
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    return dataloader




#========================== TENSOR-SAVING FUNCTION =============================
def NameAndSaveImage(image_tensor: torch.Tensor, output_path: str, image_name_batch:tuple = None, image_base_name: str = "generated_noise", bit_depth: int = 8, name_start_index:int = 0):
    """
    Save tensor image(s) as PNG with given bit depth and optional name.
    -> save as uint16 if bit_depth > 8, else uint8
    
    Args:
        image_tensor (torch.Tensor): Image tensor with shape
                                     - (N, C, H, W), batch of images
                                     - (C, H, W), single image
                                     - (H, W), grayscale image
        output_path (str): Directory where images are saved.
        image_name_batch (torch.Tnsor): if is given: name each image imege_name_batch[i], 
                                        no check if already existed
        image_base_name (str, optional): else: base name for saved image(s): '[base_name]_[n].png'
                                    If None, auto-generated as 'generated_noise_[n].png'.
        bit_depth (int, optional): Bit depth for saving (default=8).
                                   Common values: 8 or 16.
        name_start_index (int, optional): if image_base_name is None: has to number 
                                        the default filename -> start trying with
                                        'generated_noise_[name_start_index].png'
                                        and try go upwards if this: already exists
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor must be a torch.Tensor")

    if image_tensor.ndim not in [2, 3, 4]:
        raise ValueError("image_tensor must have 2, 3, or 4 dimensions")

    # Ensure batch dimension
    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)

    N, C, H, W = image_tensor.shape

    if C not in [1, 3]:
        raise ValueError("Channel dimension must be 1 (grayscale) or 3 (RGB)")

    # Normalize to [0, 1]
    tensor_min = image_tensor.min()
    tensor_max = image_tensor.max()
    if tensor_max > tensor_min:
        image_tensor = (image_tensor - tensor_min) / (tensor_max - tensor_min)
    else:
        image_tensor = torch.zeros_like(image_tensor)  # uniform image

    # Scale to bit depth
    max_val = 2 ** bit_depth - 1
    image_tensor = (image_tensor * max_val).round().clamp(0, max_val).to(torch.uint16 if bit_depth > 8 else torch.uint8) # type: ignore to supress warning

    for i in range(N):
        img = image_tensor[i]  # (C, H, W)
        if C == 1:
            img = img.squeeze(0).cpu().numpy()
            mode = "I;16" if bit_depth > 8 else "L"
        else:
            img = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            mode = None  # let PIL handle RGB automatically

        pil_img = Image.fromarray(img, mode=mode)

        # Choose filename
        if image_name_batch:
            assert len(image_name_batch) == N, "Number of image-names must match number of images in tensor."
            fname = image_name_batch[i]
            fpath = os.path.join(output_path, fname)
        else:
            n = name_start_index # start trying with this number after default name
            while True:
                fname = f"{image_base_name}_{n}.png"
                fpath = os.path.join(output_path, fname)
                if not os.path.exists(fpath):
                    break
                n += 1

        pil_img.save(fpath, format="PNG")
        print(f"Saved: {fpath}")