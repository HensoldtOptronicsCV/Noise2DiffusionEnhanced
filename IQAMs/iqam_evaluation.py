"""
================================================================================
Model Evaluation with IQA-Metrics

This script contains the code for computing the mean image-folder scores for a 
list of IQAMs. If at leat one metric is full-reference (FR), a GT image-folder 
is needed, with pairing by file-name.

It supports Tensorboard, and a lot of IQAMs. 
See the parse_args() content for an overview of run-options and run 
"print(pyiqa.list_models())" to see all available IQAMs.

Use: python iqam_evaluation.py --images_to_eval_path "path/to/to-assess/image/folder" --gt_images_path "path/to/GT/image/folder" --name_ending "test"
================================================================================
"""


import torch
import pyiqa
from torch import amp
import numpy as np # just for reshaping a numpy array...
import sys
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
    # for dict-saving as file
import json
    # for output-naming
from datetime import datetime
import torch.nn.functional as F # for padding
from tqdm import tqdm # for progress-bar
from typing import List, Tuple, Any
    # for Dataloader-stuff : look in the 'Dataloader'-folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Dataloader')))
from Dataset_and_Dataloader import SingleTIRDataset, PairedTIRDataset, GetDataLoader, NormalizeTensor, NameAndSaveImage # type: ignore to suppress warning




def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='IQAM evaluation')
    parser.add_argument(
        '--gt_images_path', type=str, default=None, metavar='HQ',
        help='path to folder with the clean test-images (GT)'
    )
    parser.add_argument(
        '--images_to_eval_path', type=str, default=3, metavar='HQ',
        help='path to folder with the noisy test-images (noisy input)'
    )
    parser.add_argument(
        '--name_ending', type=str, default='',
        help='name-sufix to give to the related files/folders -> [YYYY.MM.DD]_[folder-name]_[name_ending]'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='batch size for evaluation'
    )
    parser.add_argument(
        '--no_tensorboard', action='store_true', default=False,
        help='forbid record training log to Tensorboard'
    )
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='disables CUDA training'
    )
    parser.add_argument(
        '--save_per_image_results', action='store_true', default=False,
        help='saves individual image scores in a .json file'
    )

    args = parser.parse_args()
    return args


# =========================   MAIN EVALUATION FUNCTION   ==================================
def evaluate_folder(folder_to_evaluate: str, metric_dict:dict, metric_name_list: list, batch_size:int, folder_GT:str=None, user_suffix:str="", no_tensorboard:bool = False, device: str = "cuda", save_per_image_results:bool = False):
    """
    Evaluate a denoising model with given IQA metrics.

    Args:
        checkpoint_path (str): Path to model checkpoint.
        clean_test_images_path (str): path to folder with clean test images (GT)
        noisy_test_images_path (str): path to folder with noisy test images (noisy input)
        metric_names (list): List of metric names (strings, e.g. ["psnr", "ssim", "lpips"]).
        test_serie_name (str): name to give to tensorboard_logs and to output .json file
        no_tensorboard (bool): if do not want to store tensorboard logs
        device (str): Device for model + metrics.

    Returns:
        dict: {metric_name: average_score_over_dataset}
    """
    if folder_GT is None:
        for metric_name in metric_name_list:
            assert metric_dict[metric_name].metric_mode == "NR", f"No Ground-Truth folder given, but {metric_name} is a full-reference method that need ground truth"
    
    # --- Initialize variables ---
    run_name = make_output_folder_name(folder_to_evaluate, user_suffix)
    bit_depth = 8
    eps = 1e-3 # for clamping the images
    normalization_transform= NormalizeTensor(2**bit_depth)
    metric_sums = {name: 0.0 for name in metric_name_list}
    metric_counts = {name: 0 for name in metric_name_list}
    biqam_result_per_image_dict = {}

    # --- Update {metric: tensorboard_indice} equivalence-dict + {metric: lower_better}-dict ---
    json_path = os.path.join(os.getcwd(), "metric_indice_dict.json") # hardcoded here so the dict is always the same
    metric_indice_dict = get_and_update_tensorboard_metric_indices(json_path, metric_name_list)
    json_path_metric_properties = os.path.join(os.getcwd(), "metric_properties_dict.json") # hardcoded here so the dict is always the same
    get_and_update_metric_properties(json_path_metric_properties, metric_name_list, metric_dict)


    if folder_GT is None:
        # case with only No-ref metrics -> no need for a paired dataloader, etc
        single_img_dataset = SingleTIRDataset(folder_to_evaluate, input_transform=normalization_transform)
        single_img_dataloader = GetDataLoader(single_img_dataset, batch_size=batch_size, shuffle=False) 
        # --- Evaluation ---
        with torch.no_grad():
            for img_batch, name_batch in tqdm(single_img_dataloader, total=len(single_img_dataloader), desc="Evaluating metrics"):
                img_batch = img_batch.to(device)
                img_batch = torch.clamp(img_batch, eps, 1 - eps)
                for image_name in name_batch:
                    biqam_result_per_image_dict[image_name] = {} # initialize dict
                # --- Evaluate metrics ---
                for metric_name, metric in metric_dict.items():
                    batch_score = evaluate_metric_on_batch(metric, metric_name, img_batch, device, GT_batch = None) # a list of floats (len=batch_size)
                    # --- accumulate sum and count
                    if len(batch_score)==len(name_batch):   # case 1 result per image
                        metric_sums[metric_name] += sum(batch_score)
                        metric_counts[metric_name] += len(name_batch)   # number of images in batch
                    elif len(batch_score)==1:               # case already meaned result -> beware of shorter batches...
                        metric_sums[metric_name] += len(name_batch) * sum(batch_score)
                        metric_counts[metric_name] += len(name_batch)   # number of images in batch
                    else: input("Interrupted bcse big pb in suming metrics. Press enter to resume nevertheless...")
                    if save_per_image_results:
                        for i,image_name in enumerate(name_batch):
                            biqam_result_per_image_dict[image_name][metric_name] = batch_score[i]

    else: # case folder_GT is not None
        paired_dataset = PairedTIRDataset(folder_to_evaluate, folder_GT, input_transform=normalization_transform)
        paired_dataloader = GetDataLoader(paired_dataset, batch_size=batch_size, shuffle=False) 
        # (batchsize 1 so the inference is not influenced by other images in the batch via batchNorm2d)
        # --- Inference & Evaluation ---
        with torch.no_grad():
            for img_batch, gt_batch, name_batch in tqdm(paired_dataloader, total=len(paired_dataloader), desc="Evaluating metrics"):
                gt_batch, img_batch = gt_batch.to(device), img_batch.to(device)
                img_batch, gt_batch = torch.clamp(img_batch, eps, 1 - eps), torch.clamp(gt_batch, eps, 1 - eps)
                for image_name in name_batch:
                    biqam_result_per_image_dict[image_name] = {} # initialize dict
                # --- Evaluate metrics ---
                for metric_name, metric in metric_dict.items():
                    batch_score = evaluate_metric_on_batch(metric, metric_name, img_batch, device, GT_batch = gt_batch) # a list of floats (len=batch_size)
                    # --- accumulate sum and count
                    if len(batch_score)==len(name_batch):   # case 1 result per image
                        metric_sums[metric_name] += sum(batch_score)
                        metric_counts[metric_name] += len(name_batch)   # number of images in batch
                    elif len(batch_score)==1:               # case already meaned result -> beware of shorter batches...
                        metric_sums[metric_name] += len(name_batch) * sum(batch_score)
                        metric_counts[metric_name] += len(name_batch)   # number of images in batch
                    else: input("Interrupted bcse big pb in suming metrics. Press enter to resume nevertheless...")
                    if save_per_image_results:
                        for i,image_name in enumerate(name_batch):
                            biqam_result_per_image_dict[image_name][metric_name] = batch_score[i]
    
    # --- Average over dataset ---
    metric_avgs = {m: metric_sums[m] / metric_counts[m] for m in metric_name_list}
    metric_avgs = {k: v.item() if torch.is_tensor(v) else v for k, v in metric_avgs.items()}

    # --- Save-Evaluation ---
    # >>> raw folder-results in json file
    json_path_avg_metric_results = os.path.join(os.getcwd(), "IQAM_evaluations-json_files", "{}.json".format(run_name))
    os.makedirs(os.path.dirname(json_path_avg_metric_results), exist_ok=True) # create if folder does not exist
    with open(json_path_avg_metric_results, "w") as file:
        json.dump(metric_avgs, file, indent=2)
    # >>> raw per-image-results in json file
    if save_per_image_results:
        json_path_perImg_metric_results = os.path.join(os.getcwd(), "IQAM_evaluations-json_files", "{}_perImg.json".format(run_name))
        with open(json_path_perImg_metric_results, "w") as file:
            json.dump(biqam_result_per_image_dict, file, indent=2)
    # >>> tensorboard
    if not no_tensorboard:
        tensorboar_log_path = os.path.join(os.getcwd(), "results-Tensorboard", run_name)
        os.makedirs(os.path.dirname(tensorboar_log_path), exist_ok=True) # create folder if does not exist
        writer = SummaryWriter(log_dir=tensorboar_log_path , comment=f"_{run_name}")
        for metric_name, metric_score in metric_avgs.items():
            writer.add_scalar("metrics/result", metric_score, metric_indice_dict[metric_name])
        # add a text-entry to tensorboard to display the "tensorboard_indice → metric_name" equivalences
        sorted_indices = sorted(metric_indice_dict.items(), key=lambda x: x[1]) # Sort the pairs by index (the dict values)
        mapping_text = [f"{index} → {metric_name} ({'↓' if metric_dict[metric_name].lower_better else '↑'})" for metric_name, index in sorted_indices] # Build mapping text
        writer.add_text("metrics/index_mapping", "\n".join(mapping_text))
        # finished writing for the evaluation of that model
        writer.close()

    print(
    f"Successfully evaluated metrics. Metric results can be found in {json_path_avg_metric_results}"
        f"{'' if no_tensorboard else f', and in {tensorboar_log_path} too.'}"
    )
    return metric_avgs
# =========================   end MAIN EVALUATION FUNCTIONS   ==============================


# =========================   HELPER FUNCTIONS   ===========================================
def evaluate_metric_on_batch(metric, metric_name, img_batch_to_evaluate, device, GT_batch = None):
    """
    Evaluates metric on all images of img_batch_to_evaluate. 
    If it is a Full-ref-metric, then the metric compares to GT_batch, 
        else the metric just evaluates on img_batch_to_evaluate.

    Output is a list on cpu, of len batch_size, regardless on how the metric evaluates on the batch ('mean', 'sum' or 'none') (according to prints)
    """
    METRICS_REQUIRING_3_CHANNELS = ['ilniqe', 'musiq', 'qalign', 'paq2piq', 'wadiqam_nr', 'cnniqa']

    # --- evaluate the metric --- 
    # preprocess if metric needs 3 channels
    if metric_name in METRICS_REQUIRING_3_CHANNELS:
        img_batch_to_evaluate = ensure_rgb(img_batch_to_evaluate)
        if GT_batch is not None:
            GT_batch = ensure_rgb(GT_batch)

    # --- some special NR metric cases --- 
    if metric_name in ['musiq', 'maniqa', 'clipiqa', 'clipiqa+', 'brisque']: #  This metrics need full 32 bit floats
        with amp.autocast(device_type=device, enabled=False):
            pred_fp32 = img_batch_to_evaluate.detach().to(torch.float32)  # force to float32
            if metric_name == 'musiq': pred_fp32 = resize_for_musiq(ensure_rgb(pred_fp32)) # fuck this metric!
            batch_score = metric(pred_fp32)
    # --- FR metrics --- 
    elif metric.metric_mode == "FR": # full-ref metric
        assert GT_batch is not None, f"{metric_name} is a Full-ref-metric that needs ground truth, but no GT_batch was given."
        batch_score = metric(img_batch_to_evaluate, GT_batch)
    # --- rest of NR metrics ---
    else: 
        batch_score = metric(img_batch_to_evaluate)
    
    # --- make sure output is a list --- 
    if torch.is_tensor(batch_score): # -> normally always the case
        batch_score = batch_score.detach().flatten().tolist()
    else:
        batch_score = np.array(batch_score).reshape(-1).tolist()

    return batch_score # this is a list (according to prints: always of len batch_size, even with loss_reduction='mean' or 'sum')


def make_output_folder_name(model_path: str, user_sufix: str) -> str:
    """
    Create an output folder name from model path, date, and user prefix.
    If model_path is a file, the model name is the file name without extension.
    If model_path is a directory, the model name is the last folder name in the path.

    Format: [model-name-or-folder-name]_evaluated[YYYY.MM.DD]_[user-sufix]
    """
    # Get date in desired format
    today_str = datetime.today().strftime("%Y.%m.%d")

    # Determine if model_path is a file or directory
    if os.path.isdir(model_path):
        # Take last folder in the path
        model_name = f"Folder_{os.path.basename(os.path.normpath(model_path))}"
    else:
        # Take file name without extension
        model_name = f"Model_{os.path.splitext(os.path.basename(model_path))[0]}"

    # Combine into final name
    folder_name = f"{model_name}_evaluated{today_str}{'_' if user_sufix else ''}{user_sufix}"
    return folder_name


def get_and_update_tensorboard_metric_indices(json_path:str, metric_name_list):
    """Updates the json file where the {metric: tensorboard_indice} equivalences
    are stored in a dict, updates teh file if a new metric is detected, 
    and finally returns the dictionnary
    
    Args:
        json_path (str): path to the json file
        metric_name_list (list of str): list of the metric names that is passed 
                                        to create the metrics with pyiqa module
    """
    # Step 1: Load existing JSON or create an empty dict if file is missing
    if os.path.isfile(json_path):
        with open(json_path, "r") as file:
            metric_indice_dict = json.load(file)
    else:
        metric_indice_dict = {}

    # Step 2: Add the entry if missing
    for metric_name in metric_name_list:
        if metric_name not in metric_indice_dict:
            max_indice = 0 if not bool(metric_indice_dict) else int(max(metric_indice_dict.values())) # initialize 0 if dict is empty
            metric_indice_dict[metric_name] = max_indice + 1 # add entry

            # Step 3: Save the updated dictionary back
            with open(json_path, "w") as file:
                json.dump(metric_indice_dict, file, indent=2)

    return metric_indice_dict

def get_and_update_metric_properties(json_path:str, metric_name_list:List, metrics:Any):
    """Updates the json file where the {metric_name: lower_is_better_bool}
    are stored in a dict, updates the file if a new metric is detected, 
    and finally returns the dictionnary
    
    Args:
        json_path (str): path to the json file
        metric_name_list (list of str): list of the metric names that is passed 
                                        to create the metrics with pyiqa module
        metrics: either a dictionnary{metric_name: metric} or a list 
                 corresponding to metric_name_list
    """
    # Step 1: Load existing JSON or create an empty dict if file is missing
    if os.path.isfile(json_path):
        with open(json_path, "r") as file:
            metric_property_dict = json.load(file)
    else:
        metric_property_dict = {}

    # Step 2: Add the entry if missing
    for metric_name_idx in range(len(metric_name_list)):
        metric_name = metric_name_list[metric_name_idx]
        if metric_name not in metric_property_dict:
            if type(metrics) is dict:
                metric_property_dict[metric_name] = {'lower_better': metrics[metric_name].lower_better, 'score_range':metrics[metric_name].score_range} # add entry
            else:
                assert len(metric_name_list) == len(metrics), "The metrics argument is not a dict, but has different length from the metric_name_list, but must be same length to match names and metrics."
                metric_property_dict[metric_name] = {'lower_better': metrics[metric_name_idx].lower_better, 'score_range':metrics[metric_name_idx].score_range} # add entry
            # Step 3: Save the updated dictionary back
            with open(json_path, "w") as file:
                json.dump(metric_property_dict, file, indent=2)
    return metric_property_dict


def resize_for_musiq(x, patch_size=32):
    """hepler function to pad a tensor to have shape divisible by 32x32 patches"""
    N, C, H, W = x.shape
    H_pad = (patch_size - H % patch_size) % patch_size
    W_pad = (patch_size - W % patch_size) % patch_size
    return F.pad(x, (0, W_pad, 0, H_pad), mode="reflect")

def ensure_rgb(x):
    """ x: (N,1,H,W) → (N,3,H,W) """
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    return x

def get_best_device(no_cuda=False):
    """
    Returns cuda if available and cpu else + print info of what is being used
    and asks for user-confimation if cpu is about to be used
    """
    device = ('cuda' if torch.cuda.is_available() and not no_cuda else 'cpu')
    
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("ATTENTION: You are runing this code on CPU, which will take much longer than on GPU")
        resp = input("Are you sure you want to run on CPU? \nPress [y] if yes, anything else to abort")
        assert resp == 'y', "Aborting computing on CPU"

    return device
# =========================   end HELPER FUNCTIONS   =======================================



# =====================================   MAIN   ===========================================
if __name__ == "__main__":
    
    use_argparser_flag = True # use parser arguments if true, manually given arguments if False
    
    if use_argparser_flag:
        args = parse_args()
        # ============ Parameters if args passed ============
        no_cuda = args.no_cuda
        folder_to_evaluate=args.images_to_eval_path
        folder_GT=args.gt_images_path 
        user_suffix=args.name_ending
        no_tensorboard=args.no_tensorboard
        batch_size = args.batch_size
        save_per_image_results = args.save_per_image_results
        device = get_best_device(no_cuda)
        # ============ End Parameters ========================   
    
    else:
        # ============ Parameters if harcoded ============
        no_cuda = False
        folder_to_evaluate = "path/to/folder/HQ"
        folder_GT = "path/to/folder/GT"
        user_suffix=""
        no_tensorboard=True
        batch_size = 8
        save_per_image_results = False
        device = get_best_device(no_cuda)
        # ============ End Parameters ============


    # --- List of pyiqa-implemented metrics to choose from ---
    ''' Possible metrics:
    ['afine_all', 'afine_all_scale', 'afine_fr', 'afine_nr', 'ahiq', 'arniqa', 'arniqa-clive', 'arniqa-csiq', 'arniqa-flive', 'arniqa-kadid', 'arniqa-live', 'arniqa-spaq', 'arniqa-tid', 'brisque', 'brisque_matlab', 'ckdn', 'clipiqa', 'clipiqa+', 'clipiqa+_rn50_512', 'clipiqa+_vitL14_512', 'clipscore', 'cnniqa', 'compare2score', 'cw_ssim', 'dbcnn', 'deepdc', 'dists', 'entropy', 'fid', 'fid_dinov2', 'fsim', 'gmsd', 'hyperiqa', 'ilniqe', 'inception_score', 'laion_aes', 'liqe', 'liqe_mix', 'lpips', 'lpips+', 'lpips-vgg', 'lpips-vgg+', 'mad', 'maniqa', 'maniqa-kadid', 'maniqa-pipal', 'ms_ssim', 'msswd', 'musiq', 'musiq-ava', 'musiq-paq2piq', 'musiq-spaq', 'nima', 'nima-koniq', 'nima-spaq', 'nima-vgg16-ava', 'niqe', 'niqe_matlab', 'nlpd', 'nrqm', 'paq2piq', 'pi', 'pieapp', 'piqe', 'psnr', 'psnry', 'qalign', 'qalign_4bit', 'qalign_8bit', 'qualiclip', 'qualiclip+', 'qualiclip+-clive', 'qualiclip+-flive', 'qualiclip+-spaq', 'sfid', 'ssim', 'ssimc', 'stlpips', 'stlpips-vgg', 'topiq_fr', 'topiq_fr-pipal', 'topiq_iaa', 'topiq_iaa_res50', 'topiq_nr', 'topiq_nr-face', 'topiq_nr-face-v1', 'topiq_nr-flive', 'topiq_nr-spaq', 'topiq_nr_swin-face', 'tres', 'tres-flive', 'unique', 'uranker', 'vif', 'vsi', 'wadiqam_fr', 'wadiqam_nr']'''
    

    metric_name_list = ['psnr', 'ssim', 'lpips'] # TODO: choose the metrics you want to use
    metric_dict = {name: pyiqa.create_metric(name, device=device, loss_reduction='none') for name in metric_name_list} 

    # --- Evaluation ---
    metric_args = evaluate_folder(folder_to_evaluate = folder_to_evaluate, 
                                metric_dict = metric_dict, 
                                metric_name_list = metric_name_list,
                                batch_size = batch_size,
                                folder_GT = folder_GT, 
                                user_suffix = "", 
                                no_tensorboard = no_tensorboard, 
                                device = device,
                                save_per_image_results = save_per_image_results)
    
    print("Evaluation-results for {}: \t{}".format(folder_to_evaluate, metric_args))



