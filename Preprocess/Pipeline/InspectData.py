import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
import traceback
from collections import Counter
import warnings
from tqdm import tqdm
from typing import Optional, Dict, Any, List

from Preprocess.Pipeline.config import CACHE_DIR

# --- Filter specific FutureWarning from torch.load ---
warnings.filterwarnings("ignore", category=FutureWarning, message=".*You are using `torch.load` with `weights_only=False`.*")

# --- Project Config Import ---
try:
    from Preprocess.Pipeline import config as project_config
except ImportError:
    project_config = None

# --- Import the Dataset Class ---
try:
    from Preprocess.Pipeline.Encapsulation.CachedTensorDataset import CachedTensorDataset
except ImportError as e:
    print(f"Error: Could not import CachedTensorDataset: {e}")
    print("Please ensure CachedTensorDataset is accessible.")
    exit()

# --- Custom Collate Function (remains the same) ---
def collate_wrapper(batch):
    """ Filters out None samples and manually collates label dicts if present. """
    filtered_batch = [item for item in batch if item is not None]
    if not filtered_batch: return None
    try:
        tensors = [item[0] for item in filtered_batch]
        labels = [item[1] for item in filtered_batch]
    except (IndexError, TypeError) as e:
         print(f"Error: Invalid item structure in batch: {e}. Expected (tensor, label). First item: {filtered_batch[0]}")
         return None
    try:
        collated_tensors = torch.stack(tensors, 0)
    except Exception as e:
        print(f"Error stacking tensors: {e}"); return None

    collated_labels = {}
    if labels:
        first_label = labels[0]
        if isinstance(first_label, dict):
            all_keys = set().union(*(d.keys() for d in labels if isinstance(d, dict)))
            for key in all_keys:
                collated_labels[key] = [d.get(key) for d in labels if isinstance(d, dict)]
        else:
            try:
                from torch.utils.data._utils.collate import default_collate
                collated_labels = default_collate(labels)
            except Exception as e:
                print(f"Warning: Error using default_collate for labels: {e}. Returning labels as a list.")
                collated_labels = labels
    return collated_tensors, collated_labels
# ---------------------------


# === Reusable DataLoader Creation Function (remains the same) ===
def get_dataloader(
    config_path: str,
    dataset_type: str,
    batch_size_override: Optional[int] = None,
    num_workers_override: Optional[int] = None
    ) -> Optional[DataLoader]:
    """ Creates a DataLoader based on a pipeline configuration file. """
    # --- 1. Load Configuration ---
    if not os.path.exists(config_path):
        print(f"Error [get_dataloader]: Config file not found at '{config_path}'")
        return None
    try:
        with open(config_path, 'r') as f: config = json.load(f)
    except Exception as e:
        print(f"Error [get_dataloader]: Failed to load or parse config '{config_path}': {e}")
        return None

    # --- 2. Extract Necessary Parameters ---
    config_name = config.get('config_name', os.path.splitext(os.path.basename(config_path))[0])
    pipeline_version = config.get('pipeline_version', 'unversioned')
    cache_root = config.get('cache_dir')
    if cache_root is None or not cache_root:
        if project_config and hasattr(project_config, 'CACHE_DIR'):
             cache_root = project_config.CACHE_DIR
        else:
             print(f"Error [get_dataloader]: Cache directory ('cache_dir') could not be determined from '{config_path}' or project_config.")
             return None
    if not isinstance(cache_root, str):
        print(f"Error [get_dataloader]: Determined cache_root is not a string: {cache_root}")
        return None

    # --- 3. Determine DataLoader Parameters ---
    dl_params = config.get('data_loader_params', {})
    batch_size = batch_size_override if batch_size_override is not None else dl_params.get('batch_size', 32)
    num_workers = num_workers_override if num_workers_override is not None else dl_params.get('num_workers', 0)
    pin_memory_effective = False
    try:
        pin_memory_cfg = dl_params.get('pin_memory', False)
        if pin_memory_cfg and num_workers > 0 and torch.cuda.is_available():
            pin_memory_effective = True
    except NameError: pass # torch not available
    except Exception: pass # Error checking CUDA

    # --- 4. Construct Data Path ---
    data_dir = os.path.join(cache_root, "PipelineResult", config_name, pipeline_version, dataset_type)
    if not os.path.isdir(data_dir):
         print(f"Warning [get_dataloader]: Data directory not found: {data_dir}.")
         return None

    # --- 5. Create Dataset and DataLoader ---
    try:
        dataset = CachedTensorDataset(data_dir)
        if len(dataset) == 0:
            print(f"Warning [get_dataloader]: No samples found in {data_dir}.")
            return None
        should_shuffle = (dataset_type.lower() == 'train')
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=should_shuffle, num_workers=num_workers,
                            pin_memory=pin_memory_effective, collate_fn=collate_wrapper,
                            persistent_workers=True if num_workers > 0 else False)
        return loader
    except NameError: print("Error [get_dataloader]: CachedTensorDataset class not found."); return None
    except Exception as e:
        print(f"Error [get_dataloader]: Failed to create DataLoader for {dataset_type} from {data_dir}: {e}")
        traceback.print_exc(); return None
# ==========================================


# === Updated Inspect Function ===
def inspect(
    config_path: str,
    inspect_batch_size: int
    ):
    """ Inspect the dataset generated by a pipeline configuration. """
    print(f"--- Inspecting results based on config: {config_path} ---")

    # 1. Load Config for preliminary info
    if not os.path.exists(config_path): print(f"\nError: Config file not found at '{config_path}'"); exit()
    try:
        with open(config_path, 'r') as f: config = json.load(f)
        dataset_types = config.get('dataset_types_to_process', ['Train', 'Validation', 'Test'])
        # Extract display info now for the final summary box
        display_config_name = config.get('config_name', os.path.splitext(os.path.basename(config_path))[0])
        display_pipeline_version = config.get('pipeline_version', 'unversioned')
        display_cache_root = CACHE_DIR
        if not display_cache_root and project_config and hasattr(project_config, 'CACHE_DIR'):
             display_cache_root = project_config.CACHE_DIR
        if not display_cache_root: display_cache_root = "[Cache Dir Not Found]"
        display_base_result_dir = os.path.join(display_cache_root, "PipelineResult", display_config_name, display_pipeline_version)
    except Exception as e:
        print(f"\nError reading configuration file '{config_path}': {e}"); exit()

    # --- Inspection logic using get_dataloader ---
    results = {}
    # (Loop through dataset_types, call get_dataloader, perform checks - same as previous version)
    for ds_type in dataset_types:
        print(f"\n--- Processing: {ds_type} ---")
        results[ds_type] = {'count': 'N/A', 'valid_sample_count': 0, 'x_shape': None, 'label_type': 'Unknown',
                           'distributions': {}, 'partially_corrupted_count': 0, 'fully_corrupted_count': 0}
        label_type_detected = None; all_labels_for_dist = []
        # 1. Attempt load first sample
        print("Attempting to load first valid sample...")
        first_sample_loaded = False
        temp_loader = get_dataloader(config_path, ds_type, batch_size_override=1, num_workers_override=0)
        if temp_loader is None:
             print(f"  Failed to create DataLoader for {ds_type}. Skipping."); results[ds_type]['count'] = 0; continue
        results[ds_type]['count'] = len(temp_loader.dataset) if temp_loader.dataset else 0
        if results[ds_type]['count'] == 0: print(f"  Dataset for {ds_type} is empty."); continue
        try:
            first_valid_batch = next(iter(temp_loader), None)
            if first_valid_batch is not None:
                 x_batch_sample, y_batch_sample_collated = first_valid_batch
                 x_sample = x_batch_sample[0]; y_sample = None
                 if isinstance(y_batch_sample_collated, dict): y_sample = {k: v[0] for k, v in y_batch_sample_collated.items() if v}
                 elif isinstance(y_batch_sample_collated, (list, torch.Tensor)) and len(y_batch_sample_collated) > 0: y_sample = y_batch_sample_collated[0]
                 else: y_sample = y_batch_sample_collated
                 results[ds_type]['x_shape'] = str(x_sample.shape)
                 if isinstance(y_sample, dict):
                     if y_sample.get('engagement_string') is not None: label_type_detected = 'engagenet'
                     elif y_sample.get('engagement_numeric') is not None: label_type_detected = 'daisee'
                     else: label_type_detected = 'unknown_dict'
                 elif isinstance(y_sample, str): label_type_detected = 'engagenet_string'
                 else: label_type_detected = f'unknown ({type(y_sample).__name__})'
                 results[ds_type]['label_type'] = label_type_detected
                 print(f"  First valid sample loaded. Type: {label_type_detected}, Shape: {x_sample.shape}")
                 first_sample_loaded = True
            else: print("  Warning: First sample DataLoader did not yield any valid batches.")
        except Exception as e: print(f"  Error loading/inspecting first sample: {e}"); traceback.print_exc()
        # 2. Iterate full DataLoader
        can_calculate_dist = first_sample_loaded and label_type_detected not in ['Unknown', 'unknown_dict', 'unknown_non_dict', 'unknown (NoneType)', 'unknown (type)']
        if not first_sample_loaded: print("Skipping full iteration as first sample failed.")
        elif not can_calculate_dist: print(f"Skipping full iteration due to unknown label type: {label_type_detected}")
        else:
            print(f"Iterating through {ds_type} DataLoader (Batch Size: {inspect_batch_size}) for full inspection...")
            valid_samples_processed, partially_corrupted, fully_corrupted = 0, 0, 0
            dataloader = get_dataloader(config_path, ds_type, batch_size_override=inspect_batch_size, num_workers_override=0)
            if dataloader is None: print(f"  Failed to create main DataLoader for {ds_type}.")
            else:
                try:
                    data_iterator = tqdm(dataloader, desc=f"Inspecting {ds_type}", leave=True, unit="batch")
                    for batch in data_iterator:
                        if batch is None: continue
                        x_batch, y_batch_collated = batch; batch_size_actual = x_batch.shape[0]
                        valid_samples_processed += batch_size_actual
                        # Corruption Check
                        for i in range(batch_size_actual):
                            x_sample = x_batch[i]; num_frames = x_sample.shape[0]; corrupted_frame_count = 0
                            if num_frames == 0: fully_corrupted += 1; continue
                            for frame_idx in range(num_frames):
                                if torch.sum(x_sample[frame_idx]).item() == 0: corrupted_frame_count += 1 # Adjust check if needed
                            if corrupted_frame_count == num_frames: fully_corrupted += 1
                            elif corrupted_frame_count > 0: partially_corrupted += 1
                        # Collect Labels
                        if label_type_detected == 'engagenet' or label_type_detected == 'engagenet_string':
                            labels_to_add = y_batch_collated.get('engagement_string', []) if isinstance(y_batch_collated, dict) else (y_batch_collated if isinstance(y_batch_collated, list) else [])
                            all_labels_for_dist.extend([lbl for lbl in labels_to_add if lbl is not None])
                        elif label_type_detected == 'daisee':
                            keys_to_extract = ['engagement_numeric', 'boredom_numeric', 'confusion_numeric', 'frustration_numeric']
                            label_data = {k: y_batch_collated.get(k, [None]*batch_size_actual) for k in keys_to_extract}
                            for i in range(batch_size_actual): all_labels_for_dist.append({k: label_data[k][i] for k in keys_to_extract})
                    # Store results
                    results[ds_type]['valid_sample_count'] = valid_samples_processed; results[ds_type]['partially_corrupted_count'] = partially_corrupted; results[ds_type]['fully_corrupted_count'] = fully_corrupted
                except Exception as e: print(f"\n  Error during full DataLoader iteration for {ds_type}: {e}"); traceback.print_exc()
                # 3. Calculate Distributions
                if all_labels_for_dist:
                    print("Calculating label distributions..."); dist_results = {}
                    if label_type_detected == 'engagenet' or label_type_detected == 'engagenet_string':
                        string_labels = [str(lbl) for lbl in all_labels_for_dist if lbl is not None]
                        if string_labels: dist_results['engagement_string'] = Counter(string_labels)
                    elif label_type_detected == 'daisee':
                        for key in ['engagement_numeric', 'boredom_numeric', 'confusion_numeric', 'frustration_numeric']:
                            valid_numeric_labels = [d.get(key) for d in all_labels_for_dist if isinstance(d, dict) and d.get(key) is not None]
                            if valid_numeric_labels: dist_results[key] = Counter(valid_numeric_labels)
                    results[ds_type]['distributions'] = dist_results
                else: print("  No valid labels collected for distribution.")

    # --- Final Summary ---
    print("\n\n" + "="*30)
    print("--- Inspection Summary ---")
    print(f"Config File: {config_path}")
    print(f"Results Base Directory (Calculated): {display_base_result_dir}")
    print(f"(Config Name: {display_config_name}, Version: {display_pipeline_version})")
    print("="*30)
    for ds_type, info in results.items():
        print(f"\n{ds_type}:")
        print(f"  Potential Files Found: {info['count']}")
        if info['valid_sample_count'] > 0:
            loaded_count = info['valid_sample_count']
            print(f"  Samples Successfully Loaded by DataLoader: {loaded_count}")
            print(f"  Sample X Shape: {info['x_shape']}")
            print(f"  Detected Label Type: {info['label_type']}")
            print(f"  Corruption Summary:")
            print(f"    - Fully Corrupted (all frames invalid): {info['fully_corrupted_count']} ({info['fully_corrupted_count']/loaded_count:.1%} of loaded)")
            print(f"    - Partially Corrupted (some frames invalid): {info['partially_corrupted_count']} ({info['partially_corrupted_count']/loaded_count:.1%} of loaded)")
            if info['distributions']:
                print("  Label Distributions:")
                for label_key, counts in info['distributions'].items():
                    try: sorted_counts = sorted(counts.items(), key=lambda item: float(item[0]))
                    except ValueError: sorted_counts = sorted(counts.items())
                    dist_parts = [f"{value}: {count}" for value, count in sorted_counts]
                    print(f"    {label_key}: " + ", ".join(dist_parts))
            else: print("  (No distribution calculated or collected)")
        elif info['count'] == 0: print("  (Directory empty or dataset could not be initialized)")
        elif info['count'] > 0 : print("  (Dataset initialized, but could not load any valid samples using DataLoader)")
        else: print("  (Directory not found or error during initial scan)")
    print("\n" + "-" * 30)

    # --- NEW: Message Box for Path Structure ---
    box_width = 65
    print("\n" + "+" + "-" * (box_width-2) + "+")
    print("|" + " Output .pt File Path Structure".center(box_width-2) + "|")
    print("+" + "-" * (box_width-2) + "+")
    print("|" + " " * (box_width-2) + "|")
    print("| " + "<cache_root>/".ljust(box_width-3) + "|")
    print("|     " + "PipelineResult/".ljust(box_width-7) + "|")
    print("|         " + "<config_name>/".ljust(box_width-11) + "|")
    print("|             " + "<pipeline_version>/".ljust(box_width-15) + "|")
    print("|                 " + "<dataset_type>/  (e.g., Train)".ljust(box_width-19) + "|")
    print("|                     " + "<subject_name>/  (Subfolder)".ljust(box_width-23) + "|")
    print("|                         " + "<clip_folder>_<pipeline_version>.pt".ljust(box_width-27) + "|")
    print("|" + " " * (box_width-2) + "|")
    print("+" + "-" * (box_width-2) + "+")
    print("| Notes:".ljust(box_width-2) + "|")
    print("| - <...> placeholders derived from config/data.".ljust(box_width-2) + "|")
    print("| - <subject_name> is used as the subfolder (TensorSavingStage).".ljust(box_width-2) + "|")
    print("| - Check TensorSavingStage & _get_expected_cache_path methods.".ljust(box_width-2) + "|")
    print("+" + "-" * (box_width-2) + "+")
    # ------------------------------------------


# Example of how you would call this from your run.py:
if __name__ == "__main__":
    print("--- Running inspect function directly for demonstration ---")
    # Assume config file exists at this relative path for the example
    example_config_path = "../Pipelines/configs/ENGAGENET_10fps_quality95_randdist.json" # Adjust relative path if needed
    example_batch_size = 32

    if os.path.exists(example_config_path):
         inspect(example_config_path, example_batch_size)
    else:
         print(f"Example config file not found at: {example_config_path}")