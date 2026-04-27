import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '../dataset/'))
sys.path.append(os.path.join(dir_name, '..'))

import utils
import math

from skimage import img_as_ubyte

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../TIR-Noise-Generator')))
from Dataset_and_Dataloader import SingleTIRDataset, GetDataLoader, NormalizeTensor, NameAndSaveImage

parser = argparse.ArgumentParser(description='Image denoising evaluation on SIDD')
parser.add_argument('--input_dir', default='/data1/wangzd/datasets/denoising/sidd_val/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/data1/wangzd/uformer_cvpr/results_release/denoising/SIDD/Uformer_B/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='/home/go69niz/Masterarbeit-Codebase/Reference-Models/Uformer/logs/denoising/sidd/Uformer_B_0706/models/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Uformer_B', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
parser.add_argument('--input_bit_depth', type=int, default=8, help='bit depth of input images')
parser.add_argument('--output_bit_depth', type=int, default=8, help='bit depth of output images')
parser.add_argument('--grayscale_output', action='store_true', help='convert output to grayscale before saving')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if not torch.cuda.is_available():
    print("WARNING: CUDA not available, running on CPU. This will be slow!")

model_restoration= utils.get_arch(args)

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()

def expand2square(timg,factor=16.0):
    """Expand image to square by padding, ensuring dimensions are multiples of factor"""
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
    
    return img, mask


# Prepare dataset and dataloader
normalization_transform = NormalizeTensor(2 ** args.input_bit_depth)
inference_dataset = SingleTIRDataset(args.input_dir, input_transform=normalization_transform)
inference_loader = GetDataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False)

print(f"Processing {len(inference_dataset)} images from {args.input_dir}")
print(f"Saving results to {args.result_dir}")

# Process images
with torch.no_grad():
    for step, (noisy_image_batch, associated_name_batch) in enumerate(inference_loader):
        # Move to GPU
        noisy_image_batch = noisy_image_batch.cuda()
        
        # Get original dimensions
        _, _, h, w = noisy_image_batch.shape
        
        # Expand to square with padding (required for the model)
        noisy_padded, mask = expand2square(noisy_image_batch, factor=128)
        
        # Run model
        restored_padded = model_restoration(noisy_padded)
        
        # Remove padding using mask
        restored = torch.masked_select(restored_padded, mask.bool()).reshape(1, 3, h, w)
        
        # Clamp to valid range
        restored = torch.clamp(restored, 0., 1.)
        
        # Convert to grayscale if requested
        if args.grayscale_output:
            # Standard RGB to grayscale conversion: 0.299*R + 0.587*G + 0.114*B
            weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).type_as(restored)
            restored = (restored * weights).sum(dim=1, keepdim=True)
            # Expand back to 3 channels for saving (grayscale with same value in R,G,B)
            #restored = restored.repeat(1, 3, 1, 1)
        
        # Save image
        NameAndSaveImage(restored, args.result_dir, associated_name_batch, bit_depth=args.output_bit_depth)

print("Inference complete!")








"""
LEGACY --> look at that if you have problems

# Process data
# FUCK!!! AGAIN THAT SHITTY .mat FORMAT -> NEED TO REVRITE INFERENCE FUNCTION!!!
filepath = os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat')
img = sio.loadmat(filepath)
Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
Inoisy /=255. # -> now: numpy array in [0,1]
print(Inoisy.shape)
restored = np.zeros_like(Inoisy) # what is this original image?... patches of same image? or 
with torch.no_grad():
    for i in tqdm(range(40)):
        for k in range(32):
            noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda() # 1 image only
            _, _, h, w = noisy_patch.shape # -> now: probablement CxHxW, torch tensor
            noisy_patch, mask = expand2square(noisy_patch, factor=128) # ???
            restored_patch = model_restoration(noisy_patch) # restoration
            restored_patch = torch.masked_select(restored_patch,mask.bool()).reshape(1,3,h,w)
            restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            restored[i,k,:,:,:] = restored_patch

            save_file = os.path.join(result_dir_img, '%04d_%02d.png'%(i+1,k+1)) # it saves patches, motherfucker -> see how can display results given by training with sidd...
            utils.save_img(save_file, img_as_ubyte(restored_patch))

# save denoised data
sio.savemat(os.path.join(result_dir_mat, 'Idenoised.mat'), {"Idenoised": restored,})


#====================  MY CODE  ======================
input_bit_depth = 8
normalization_transform= NormalizeTensor(2**input_bit_depth)

lq_inference_image_path = args.input_dir
lq_inference_dataset = SingleTIRDataset(lq_inference_image_path,input_transform=normalization_transform)
inference_data_loader = GetDataLoader(lq_inference_dataset, batch_size=1, shuffle=False)

for step, (noisy_image_batch, associated_name_batch) in enumerate(inference_data_loader):
    # make prediction = forward path
    unet_prediction = model_restoration(noisy_image_batch)

    unet_prediction = torch.clamp(unet_prediction, 0., 1.) # to avoid resizings by NameAndSaveImage
    NameAndSaveImage(unet_prediction, output_folder, associated_name_batch, bit_depth = output_bit_depth)
    """