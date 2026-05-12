'''
================================================================================
This file contains the scripts to affectively load, degrade and store/visualize 
images with the TIR_noise_generator and adequate folder names.
================================================================================
'''



from TIR_noise_generator import SampleNoise, GetNoiseSamplingParams, GetNoiseSamplingParamsLighterNoise
from Dataset_and_Dataloader  import SingleTIRDataset, GetDataLoader, NameAndSaveImage, NormalizeTensor, RandomCrop, random_crop_batch

import matplotlib.pyplot as plt # for image display at end
import torch
from torchvision import transforms # for image normalization


def NoiseNamingSuffixCreationTrain(base_name:str , noise_params:dict) -> str :
    """
    This function creates the suffix for naming noise-samples: it returns
            "[base_name]_G{},{}-LC{},{}-L{},{}-C{},{}-HN{},{}" 
    with inside the breakets the values of the ranges for respectively 
    Gaussian, Line/Column, Line, Column and Hill-Noise.

    Args:
        base_name: the name before the range-values-suffix
        noise_params: the dictionnary with all noise-parameters, 
            following the output format of GetNoiseSamplingParams()
    """

    string_name = "{}_G{},{}-LC{},{}-L{},{}-C{},{}-HN{},{}".format(base_name, int(noise_params['gaussian_noise_std_range'][0] *255 ), int(noise_params['gaussian_noise_std_range'][1] *255 ), int(noise_params['row_col_noise_std_range'][0] *255 ), int(noise_params['row_col_noise_std_range'][1] *255 ), int(noise_params['row_noise_std_range'][0] *255 ), int(noise_params['row_noise_std_range'][1] *255 ), int(noise_params['col_noise_std_range'][0] *255 ), int(noise_params['col_noise_std_range'][1] *255 ), int(noise_params['hill_noise_std_range'][0] *255 ), int(noise_params['hill_noise_std_range'][1] *255 ))
    return string_name


def NoiseNamingSuffixCreationTest(base_name:str , noise_params:dict) -> str :
    """
    This function creates the suffix for naming noise-samples: it returns
            "[base_name]_G{}-LC{}-L{}-C{}-HN{}" 
    with inside the breakets the values of the fix values (not the ranges) 
    for respectively Gaussian, Line/Column, Line, Column and Hill-Noise.

    Args:
        base_name: the name before the fix-values-suffix
        noise_params: the dictionnary with all noise-parameters, 
            following the output format of GetNoiseSamplingParams()
    """
    string_name = "{}_G{}-LC{}-L{}-C{}-HN{}".format(base_name, int(noise_params['gaussian_noise_std_value'] *255 ), int(noise_params['row_col_noise_std_value'] *255 ), int(noise_params['row_noise_std_value'] *255 ), int(noise_params['col_noise_std_value'] *255 ), int(noise_params['hill_noise_std_value'] *255 ))
    return string_name




#============================================ MAIN =============================
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bit_depth = 8
    normalization_transform= NormalizeTensor(2**bit_depth)
    # crop_tranform = RandomCrop(crop_height=512, crop_width=640)
    composed_transform = transforms.Compose([normalization_transform])


    #============== for visualizing degraded images ======================================
    """
    hq_dataset = SingleTIRDataset("path/to/folder/dataset/images", input_transform=normalization_transform)
    hq_dataloader = GetDataLoader(hq_dataset, batch_size=64, shuffle=False) # shuffle makes you see always same images or randomly chosen ones
    # get lq_image batch
    hq_image_batch, associated_name_batch = next(iter(hq_dataloader)) # access a random set of hq_images as batch (tensor of size batchsize)

    # sanity checks
    print("hq_image_batch from dataloader - size : ", hq_image_batch.size())
    print("associated_name_batch from dataloader - size : ", associated_name_batch)
    
    plt.imshow(hq_image_batch.cpu().numpy()[0, 0, :, :], cmap='gray',
            vmin=hq_image_batch.min().item(), vmax=hq_image_batch.max().item())
    plt.colorbar()
    plt.show()

    # generate noise
    #noise_params = GetNoiseSamplingParams()
    noise_params = GetNoiseSamplingParamsLighterNoise()
    N,C,H,W = hq_image_batch.size()
    noisy_images = SampleNoise(N, C, H, W, noise_params, device, mode="train") # first: just noises
    assert noisy_images.size() == hq_image_batch.size()
    noisy_images += hq_image_batch 

    #watch result-sample
    plt.imshow(noisy_images.cpu().numpy()[0, 0, :, :], cmap='gray',
            vmin=noisy_images.min().item(), vmax=noisy_images.max().item())
    plt.colorbar()
    plt.show()

    # and save it
    output_path = "path/to/target/folder"
    NameAndSaveImage(noisy_images, output_path, image_name_batch = associated_name_batch)
    """
    
    #============== for corrupting all images of a folder ======================================

    noise_params = GetNoiseSamplingParamsLighterNoise()
    output_path = "path/to/target/folder/noisy/counterparts"
    input_path = "path/to/folder/images/clean"
    hq_dataset = SingleTIRDataset(input_path,input_transform=normalization_transform)
    hq_dataloader = GetDataLoader(hq_dataset, batch_size=1, shuffle=False)

    # corrupt batch-wise all images of the folder
    for hq_image_batch, associated_name_batch in hq_dataloader:
        hq_image_batch = hq_image_batch.to(device)
        #cropped_batch = random_crop_batch(hq_image_batch)

            # generate noise
        N,C,H,W = hq_image_batch.size()
        noisy_images = SampleNoise(N, C, H, W, noise_params, device, mode="train").to(device)
        noisy_images += hq_image_batch 
            # and save it (is normalized before saving)
        NameAndSaveImage(noisy_images, output_path, image_name_batch = associated_name_batch)
    