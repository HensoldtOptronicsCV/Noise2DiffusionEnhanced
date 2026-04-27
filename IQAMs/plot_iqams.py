"""
================================================================================
Plot json dict with metric evaluation results.

See iqam_evaluation.py for the structure of the dict: it is the one dumped in the 
default json file containing the average IQAMs computed for a specific folder.
================================================================================
"""



import os
import json
import matplotlib.pyplot as plt
import numpy as np
import copy # for copying the x-plots

def plot_json_metrics_raw(folder_path: str):
    """
    Reads all .json files in a folder, extracts metric dictionaries, 
    and plots the metrics (keys = x-axis, values = y-axis).
    
    The label for each curve is the part of the filename before "_evaluated".
    """
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    if not json_files:
        print("No JSON files found in the provided folder.")
        return

    plt.figure(figsize=(12, 6))

    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue

        # Extract filename label
        label = json_file.split("_evaluated")[0]

        # Load metric ordering from external JSON file
        try:
            with open("./metric_indice_dict.json", "r") as f:
                metric_order = json.load(f)
        except Exception as e:
            metric_order = {"psnr": 1, "ssim": 2, "lpips": 3, "lpips-vgg": 4, "lpips-vgg+": 5, "musiq": 6, "maniqa": 7, "maniqa-pipal": 8, "clipiqa": 9, "clipiqa+": 10, "clipiqa+_vitL14_512": 11, "niqe": 12, "brisque": 13, "dists": 14}

        # Sort metrics according to the given ordering
        metrics = sorted(data.keys(), key=lambda m: metric_order.get(m, float('inf')))
        values = [data[m] for m in metrics]

        plt.plot(metrics, values, marker="o", label=label)

    plt.title("Comparison of Metrics Results Across Models")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_json_metrics_scaled(folder_path: str, value_range=(0, 10)):
    """
    Reads all .json files in a folder, extracts metric dictionaries, 
    scales each metric across all files to a given value range, and plots them.
    
    - If a metric is 'lower is better', its values are inverted before scaling.
    - Scaling is done globally per metric based on min/max across all files.
    - The label for each curve is the part of the filename before "_evaluated".
    """
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    if not json_files:
        print("No JSON files found in the provided folder.")
        return

    # Try to load metric order
    metric_indice_file = "./metric_indice_dict.json"
    if os.path.isfile(metric_indice_file) and os.path.getsize(metric_indice_file) > 0:
        with open(metric_indice_file, "r") as f:
            metric_order = json.load(f)
    else:
        metric_order = {
            "psnr": 1, "ssim": 2, "lpips": 3, "lpips-vgg": 4, "lpips-vgg+": 5,
            "musiq": 6, "maniqa": 7, "maniqa-pipal": 8, "clipiqa": 9,
            "clipiqa+": 10, "clipiqa+_vitL14_512": 11, "niqe": 12,
            "brisque": 13, "dists": 14}
    # Open Metric-property file -> {metric_name: {'lower_better': True/False, 'score_range': "~0, ~150"} }
    json_path_metric_properties = os.path.join(os.getcwd(), "metric_properties_dict.json")
    with open(json_path_metric_properties, "r") as file:
        metric_property_dict = json.load(file)

    # --- Pass 1: Gather all values per metric ---
    all_metrics_values = {m: [] for m in metric_order.keys()}
    data_per_file = {}

    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        with open(json_path, "r") as f:
            data = json.load(f)
        label = json_file.split("_evaluated")[0]
        data_per_file[label] = data

        # Record values (inverted if lower is better)
        for metric, value in data.items():
            if metric in all_metrics_values:
                val = -value if metric_property_dict[metric]['lower_better'] else value
                all_metrics_values[metric].append(val)

    # --- Compute min/max per metric ---
    metric_minmax = {}
    for metric, vals in all_metrics_values.items():
        if vals: metric_minmax[metric] = (min(vals), max(vals))

    vmin, vmax = value_range # global range to be scaled to
    plt.figure(figsize=(12, 8))
    metrics = sorted(all_metrics_values.keys(), key=lambda m: metric_order.get(m, float('inf')))
    metrics_legend_with_scale = copy.deepcopy(metrics)
    for idx, metric in enumerate(metrics):
        metric_min, metric_max = metric_minmax[metric]
        metrics_legend_with_scale[idx] += f" | (x-{metric_min:.2f})/{(metric_max-metric_min):.2f}"


    # --- Pass 2: Plot scaled metrics per file ---
    for label, data in data_per_file.items():
        scaled_values = []

        for metric in metrics:
            if metric not in metric_minmax:
                scaled_values.append(np.nan)
                continue

            val = -data[metric] if metric_property_dict[metric]['lower_better'] else data[metric]
            min_val, max_val = metric_minmax[metric]

            # Avoid division by zero
            if max_val == min_val:
                scaled = (vmax + vmin) / 2 # fixed value on that metric for all runs
            else:
                scaled = vmin + (val - min_val) * (vmax - vmin) / (max_val - min_val)
            scaled_values.append(scaled)

        plt.plot(metrics_legend_with_scale, scaled_values, marker="o", label=label)

    plt.title(f"Scaled Comparison of Metrics Across Models ({vmin}-{vmax})")
    plt.xlabel("Metric")
    plt.ylabel("Scaled Value")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    json_results_path = r'.\results-json'
    plot_json_metrics_scaled(json_results_path, value_range=(0, 10))