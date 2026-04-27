



# **Self-supervised Diffusion-guided Hallucination-free Thermal Infrared Image Denoising (IEEE2026)**
 Félix Hazebrouck<sup>1,2</sup>, Alexander Schock-Schmidtke<sup>1,3</sup>, Norbert Stuhrmann<sup>2</sup>, Johannes Fottner<sup>1</sup>, Michael Teutsch<sup>2</sup>  
<small><sup>1</sup> Technical University of Munich (TUM), Germany
<sup>2</sup> HENSOLDT, Germany, <sup>3</sup> digital workbench, Germany</small>  

TODO: kann man nur IEEE2026 schreiben oder lieber den Workshop nennen? +  ev. add link to personal GitHub like Uformer for example.

 <br/><br/>
<div align="center">
<table>
<tr>

<td align="center" height="60">
<a href="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTjydBgaiy1TPR3G0LZPDffdruvfqFSfb37lA&s">
<img src="./README_figures/button_paper.png" height="50"/>
</a>
</td>

<td align="center" height="60">
<a href="https://github.com/HensoldtOptronicsCV/Noise2DiffusionEnhanced">
<img src="./README_figures/button_github.png" height="50"/>
</a>
</td>

<td align="center" height="60">
<a href="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTjydBgaiy1TPR3G0LZPDffdruvfqFSfb37lA&s">
<img src="./README_figures/button_huggingface.png" height="50"/>
</a>
</td>

</tr>
</table>
</div>

![Graphical Abstract](./README_figures/display_image.png)
*Our approach uses diffusion-based image enhancement and realistic TIR image degradation to generate image pairs for supervised learning (a) and leverages remarkable visual quality of diffusion models (c) without suffering from hallucinations (d-e).*  




<br/><br/>
## Repository Overview

[1. *HDRT-TIR-DE* Dataset](#hdrt_tir_de)  
[2. *Noise2DiffusionEnhanced* Pretrained Denoising Network](#Noise2DiffusionEnhanced)  
[3. TIR Sensor-Noise Generator](#noise_generator)  
[4. IQAM Evaluation](#iqam)

<br/><br/>
## 1. *HDRT-TIR-DE* Dataset <a name="hdrt_tir_de"></a> 

The *HDRT-TIR-DE dataset* is a large-scale reference thermal infrared single-image dataset designed to serve as clean reference for self-supervised training schemes.  
 In the name *HDRT-TIR-DE*, *TIR* stands for *Thermal InfraRed* and *DE* for *Diffusion-Enhanced*, while *HDRT* is the name of the dataset the HDRT-TIR-DE build upon.

 The dataset can be downloaded on HuggingFace [here](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTjydBgaiy1TPR3G0LZPDffdruvfqFSfb37lA&s), along with the train, validation and test splits.  
 The files and content of the dataset are further described there, as well as in the [supplementary meterial](todo) of our work.

As described in our [paper](), the dataset builds upon the thermal part of the [HDRT dataset](https://huggingface.co/datasets/jingchao-peng/HDRTDataset), introduced in [[1]](#1), and which was enhanced with an image-restoration diffusion model ([StableSR](https://link.springer.com/article/10.1007/s11263-024-02168-7)) to achieve better perceptual quality than any existing real TIR dataset. Combined with the diverse scenes and high image resolution inherited from the original HDRT-TIR dataset, the HDRT-TIR-DE dataset is particularely well suited for self-supervised training of image restoration networks. The specific restoration task can be determined by the degradation model used for generating the LQ counterparts from the clean images. In our work, we placed focus on TIR sensor-noise removal.

This dataset should contribute to filling the current lack of clean TIR reference images, due to imperfections in real TIR imagers that inevitably introduce noise in every captured real TIR image.  

<br/><br/>
## 2. *Noise2DiffusionEnhanced* Pretrained Denoising Network <a name="Noise2DiffusionEnhanced"></a> 

The *Noise2DiffusionEnhanced* network is a pretrained TIR sensor-noise denoising network with [Uformer architecture](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.html). It was trained on the [HDRT-TIR-diffusion-enhanced](https://huggingface.co/datasets/Aldunitro/HDRT-TIR-diffusion-enhanced) dataset and represents an out-of-the-box working denoising network.  
This pretrained model should contribute to filling the current lack of reference denoising models for TIR single-image denoising, for direct use as well as for related research.  

This section provides a step-by-step installation guide for running inference and/or fine-tuning this denoising method.  


### 2.1) Environement Setup
For this setup, you will need to have anaconda installed on your machine and accessible through the ```conda``` command in your terminal. 

1 - Check if you have a GPU that is recognized:  
```bash
nvidia-smi
```
If you see something like
```bash
Driver Version: 535.xx
CUDA Version: 12.x
```
everything is fine.   
If this line fails, please debug the driver or add a GPU to your machine.  

2 - Clone the repository on your machine:
```bash
cd path/to/future/location/local/copy
git clone https://github.com/HensoldtOptronicsCV/Noise2DiffusionEnhanced.git
```
and go to the ```Uformer_modified``` folder inside the main folder of the local copy:
```bash
cd ./Noise2DiffusionEnhanced/Uformer_modified
```  

3 - Install the environement from the ```uformer_env_no-builds.yml``` file:
```bash
conda env create -f uformer_env_no-builds.yml
```
Possible improvement if dependency solving is slow:
```bash
mamba env create -f uformer_env_no-builds.yml
```
> If this step fails, you can also try installing the environement following the installation guide from the original [Uformer repo](https://github.com/ZhendongWang6/Uformer): the ```requirements.txt``` file is still included in this modified version of the Uformer repository, but we had to comment out the first two lines and install torch and torchvision manually to the latest version to avoid package conflicts. The versions we ended up with are specified in the ```uformer_env_no-builds.yml``` file.  


### 2.2) Download Pretrained Weights

Download the pretrained weights from the [Huggingface page](todo) in form of the ```model_best.pth``` file, and move it to ```./Uformer_modified/log/Uformer_B/models/model_best.pth``` (```./``` is the main folder of the repository).  
You can move the file to another location, the path to the weights can be set in the ```options.py``` file, the default is the path above.   

### 2.3) Run Inference or Fine-Tune the Model

Set the default variables in ```options.py``` (in ```./Uformer_modified/```) and look at the options that can be modified.

**For inference**, run :  
```bash
python test/inference_custom.py \
--input_dir path/to/noisy/images \
--result_dir path/to/result/directory \
--weights ./Uformer_modified/log/Uformer_B/models/model_best.pth \
--grayscale_output
```

**For fine-tuning the weights**, run :
```bash
python3 ./train/train_denoise.py --arch Uformer_B --dataset sidd --warmup \
--env _train01_fine-tuning \
--train_dir_HQ "path/to/folder/target/images/train" \
--val_dir_HQ "path/to/folder/target/images/val" \
--val_dir_LQ "path/to/folder/noisy/images/val" \
--save_dir ./logs/ \
--batch_size 16 --batch_size 32 --train_ps 128 --gpu '0' \
--resume --pretrain_weights "./Uformer_modified/log/Uformer_B/models/model_best.pth”
```
> :information_source NOTES for fine-tuning: 
> - Please do not touch the arguments of the first line (typically the sidd is the option to activate the part of the code I grafted the my noise-generation on, even if there is nothing effective left of the SIDD dataset).
> - The images are paired by name across the HQ- and LQ directory.
> - There is no LQ image folder, as the sensor-noise is sampled annew for each batch and added in a data-augmentation way. If you want to train on fixed noisy images, please use the ```--train_dir_LQ``` argument (also in ```options-py```).
> - You can play with the different options, but we recommend to stay close to the default option-values for reproductibility and because we did not explore the effects of variations in each argument value on training.
> - If you wish to modify the code for more freedom with your data, please notice that the main adaptions with respect to the original Uformer code were achieved in the Dataset class in ```./Uformer_modified/dataset/dataset_denoise.py```. This is probably the best graft-point for your custom data setup.  


<br/><br/>
## 3. TIR Sensor-Noise Generator <a name="noise_generator"></a> 

We are currently exploring the possibilities to publish modified code from [the original repo](https://github.com/cailijing/MDIVDnet).  
If the repo-owners refuse to add a license to their code, we will try and publish a script to add all modifications we made to their code, so you can download their code from the original repo and apply our modifications with the specific script. Let's hope they add a license...  


<br/><br/>
## 4. IQAM Evaluation <a name="iqam"></a> 

In our work, we use the Image Quality Assessement Method (IQAM) library [*pyiqa*](https://github.com/chaofengc/IQA-PyTorch) introduced in [[2]](#2) and widely used in computer vision projects.

### 4.1) Environement Setup
For this setup, you will need to have anaconda installed on your machine and accessible through the ```conda``` command in your terminal. 

1 - Check if you have a GPU that is recognized (not mandatory for some IQAMs but recommended):  
```bash
nvidia-smi
```
If you see something like
```bash
Driver Version: 535.xx
CUDA Version: 12.x
```
everything is fine.   
If this line fails, please debug the driver or add a GPU to your machine.  

2 - Clone the repository on your machine:
```bash
cd path/to/future/location/local/copy
git clone https://github.com/HensoldtOptronicsCV/Noise2DiffusionEnhanced.git
```
and go to the ```IQAMs``` folder inside the main folder of the local copy:
```bash
cd ./Noise2DiffusionEnhanced/IQAMs
```  

3 - Install the environement from the ```n2de_env_no-builds.yml``` file (:warning bif environement, see bolow):
```bash
conda env create -f n2de_env_no-builds.yml
```
Possible improvement if dependency solving is slow:
```bash
mamba env create -f n2de_env_no-builds.yml
```
> This envoronement is the project base environement. The .yml installation file was extracted from a working environement but the whole environement is overkill for this task. Therefore, you can also try and piece together a minimal environement using the imports at the beginning of the document, but we give no waranty on the output of that method.  
> If someone successfully pieces toghether a working minimal environement for the scripts of the ```IQAMs``` folder, you can paste the exported .yml file in the issues to help other users.  

### 4.2) Run IQAM Evaluation Script

Please have a look at the call-options in ```iqam_evaluation.py.py```.

**For inference**, run :  
```bash
python iqam_evaluation.py \
--images_to_eval_path path/to/folder/images/to/evaluate \
--gt_images_path path/to/folder/clean/reference/images
```
> :information_source NOTES for IQAM evaluation: 
> - per default, only PSNR, SSIM and LPIPS metrics are computed. Please pick the metrics you want to compute from the list given in the ```iqam_evaluation.py``` file in the main function at the end of the script. The use is quite intuitive.
> - If only No-Reference (NR) metrics are selected for being computed, the ```--gt_images_path``` option can be omitted or set to None.

<br/><br/>
## Citation

If you use the HDRT dataset, please cite our work:

    @article{TODO
    }




<br/><br/>
## References
<a id="1">[1]</a> 
Jingchao Peng, Thomas Bashford-Rogers, Francesco Banterle, Haitao Zhao, and Kurt Debattista. HDRT: A large-scale dataset for infrared-guided HDR imaging. *Elsevier Information Fusion*, 120, 2025

<a id="2">[2]</a> 
Chaofeng Chen and Jiadi Mo. IQA-PyTorch: PyTorch Toolbox for Image Quality Assessment, 2022

