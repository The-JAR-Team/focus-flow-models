import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
import traceback
from collections import Counter
import warnings
from tqdm import tqdm
from typing import Optional, Dict, Any, List
import textwrap

from Preprocess.Pipeline import OrchestrationPipeline

# --- Import Tabulate ---
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    print("Warning: 'tabulate' library not found. Hintonstall it using: pip install tabulate")
    TABULATE_AVAILABLE = False

# --- Central Cache Directory Import ---
try:
    from Preprocess.Pipeline.config import CACHE_DIR
    if not CACHE_DIR or not isinstance(CACHE_DIR, str):
         print("Warning: Imported CACHE_DIR from config.py is invalid.")
         CACHE_DIR = None
except ImportError:
    print("Warning: Could not import CACHE_DIR from Preprocess.Pipeline.config.")
    CACHE_DIR = None

# --- Import the Dataset Class ---
try:
    from Preprocess.Pipeline.Encapsulation.CachedTensorDataset import CachedTensorDataset
except ImportError as e:
    print(f"Error: Could not import CachedTensorDataset: {e}"); exit()

# --- Filter specific FutureWarning from torch.load ---
warnings.filterwarnings("ignore", category=FutureWarning, message=".*You are using `torch.load` with `weights_only=False`.*")


# --- Custom Collate Function ---
# (No changes needed here, assumes labels within a batch are consistent type)
def collate_wrapper(batch):
    filtered_batch = [item for item in batch if item is not None]
    if not filtered_batch: return None
    try:
        tensors = [item[0] for item in filtered_batch]
        labels = [item[1] for item in filtered_batch]
    except (IndexError, TypeError) as e: print(f"Error: Invalid item structure: {e}. Item: {filtered_batch[0]}"); return None
    try: collated_tensors = torch.stack(tensors, 0)
    except Exception as e: print(f"Error stacking tensors: {e}"); return None
    collated_labels = {}
    if labels:
        first_label = labels[0]
        if isinstance(first_label, dict):
            # Collect all keys present across dictionaries in the batch
            all_keys = set().union(*(d.keys() for d in labels if isinstance(d, dict)))
            for key in all_keys:
                # Ensure list contains values or None for all items in batch
                collated_labels[key] = [d.get(key) for d in labels if isinstance(d, dict)]
        else: # Handle non-dict labels
            try:
                from torch.utils.data._utils.collate import default_collate
                collated_labels = default_collate(labels)
            except Exception as e: print(f"Warning: Collate error: {e}"); collated_labels = labels # Fallback
    return collated_tensors, collated_labels
# ---------------------------


# === Reusable DataLoader Creation Function ===
# (No changes needed here)
def get_dataloader(config_path: str, dataset_type: str, batch_size_override: Optional[int] = None, num_workers_override: Optional[int] = None, shuffle=False, transform_pipeline: Optional[OrchestrationPipeline] = None) -> Optional[DataLoader]:
    if CACHE_DIR is None: print("Error [get_dataloader]: CACHE_DIR not available."); return None
    if not os.path.exists(config_path): print(f"Error [get_dataloader]: Config not found: '{config_path}'"); return None
    try:
        with open(config_path, 'r') as f: config = json.load(f)
    except Exception as e: print(f"Error [get_dataloader]: Failed loading config '{config_path}': {e}"); return None
    config_name = config.get('config_name', os.path.splitext(os.path.basename(config_path))[0])
    pipeline_version = config.get('pipeline_version', 'unversioned')
    cache_root = CACHE_DIR
    dl_params = config.get('data_loader_params', {}); batch_size = batch_size_override if batch_size_override is not None else dl_params.get('batch_size', 32)
    num_workers = num_workers_override if num_workers_override is not None else dl_params.get('num_workers', 0); pin_memory_effective = False
    try:
        if dl_params.get('pin_memory', False) and num_workers > 0 and torch.cuda.is_available(): pin_memory_effective = True
    except Exception: pass
    data_dir = os.path.join(cache_root, "PipelineResult", config_name, pipeline_version, dataset_type)
    if not os.path.isdir(data_dir): print(f"Warning [get_dataloader]: Data dir not found: {data_dir}."); return None
    try:
        dataset = CachedTensorDataset(data_dir, transform_pipeline=transform_pipeline, verbose_dataset=False)
        if len(dataset) == 0: print(f"Warning [get_dataloader]: No samples found in {data_dir}."); return None
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            pin_memory=pin_memory_effective, collate_fn=collate_wrapper, persistent_workers=True if num_workers > 0 else False)
        return loader
    except Exception as e: print(f"Error [get_dataloader]: Failed creation for {dataset_type}: {e}"); traceback.print_exc(); return None
# ==========================================


# === Helper Function for Formatting Distributions ===
# (No changes needed here, it already iterates through provided keys)
def format_distribution_dict(distributions: Dict[str, Counter], indent: str = " " * 4) -> str:
    output_lines = []
    if not distributions: return f"{indent}(No distribution calculated)"
    for label_key, counts in sorted(distributions.items()): # Sort keys for consistent order
        output_lines.append(f"{indent}{label_key}:")
        if not counts: output_lines.append(f"{indent}  (No counts)"); continue
        try: sorted_counts = sorted(counts.items(), key=lambda item: float(item[0]))
        except ValueError: sorted_counts = sorted(counts.items())
        dist_parts = [f"{value}: {count}" for value, count in sorted_counts]
        dist_text = ", ".join(dist_parts)
        wrapped_lines = textwrap.wrap(dist_text, width=70, subsequent_indent=f"{indent}  ", break_long_words=True)
        output_lines.extend(wrapped_lines)
    return "\n".join(output_lines)
# ==========================================


# === Updated Inspect Function ===
def inspect(
    config_path: str,
    inspect_batch_size: int
    ):
    """ Inspect the dataset generated by a pipeline configuration. """
    print(f"--- Inspecting results based on config: {config_path} ---")

    if CACHE_DIR is None: print("FATAL ERROR: CACHE_DIR unavailable."); return None
    if not os.path.exists(config_path): print(f"\nError: Config file not found: '{config_path}'"); return None

    # Load Config and basic info
    try:
        with open(config_path, 'r') as f: config = json.load(f)
        dataset_types = config.get('dataset_types_to_process', ['Train', 'Validation', 'Test'])
        display_config_name = config.get('config_name', os.path.splitext(os.path.basename(config_path))[0])
        display_pipeline_version = config.get('pipeline_version', 'unversioned')
        display_base_result_dir = os.path.join(CACHE_DIR, "PipelineResult", display_config_name, display_pipeline_version)
    except Exception as e: print(f"\nError reading config '{config_path}': {e}"); return None

    # --- Inspection logic ---
    results = {}

    for ds_type in dataset_types:
        print(f"\n--- Processing: {ds_type} ---")
        results[ds_type] = {'count': 'N/A', 'valid_sample_count': 0, 'x_shape': None, 'label_type': 'Unknown',
                           'distributions': {}, 'partially_corrupted_count': 0, 'fully_corrupted_count': 0,
                           'all_label_keys': set() } # Store all unique keys found in dict labels
        label_type_detected = None
        all_labels_for_dist = [] # Store raw labels for distribution calculation
        first_sample_loaded = False

        # 1. Attempt load first sample to detect structure
        print("Attempting to load first valid sample...")
        temp_loader = get_dataloader(config_path, ds_type, batch_size_override=1, num_workers_override=0)
        if temp_loader is None: print(f"  Failed to create DataLoader. Skipping."); results[ds_type]['count'] = 0; continue
        results[ds_type]['count'] = len(temp_loader.dataset) if temp_loader.dataset else 0
        if results[ds_type]['count'] == 0: print(f"  Dataset empty."); continue

        try:
            first_valid_batch = next(iter(temp_loader), None)
            if first_valid_batch is not None:
                 x_batch_sample, y_batch_sample_collated = first_valid_batch
                 x_sample = x_batch_sample[0]
                 y_sample = None # The actual first label value/dict

                 # Determine label structure and type from first batch's labels
                 if isinstance(y_batch_sample_collated, dict):
                     # It's dictionary type, extract first actual label dict
                     y_sample = {k: v[0] for k, v in y_batch_sample_collated.items() if v}
                     if isinstance(y_sample, dict):
                         results[ds_type]['all_label_keys'].update(y_sample.keys()) # Add keys from first sample
                         # Basic type detection based on keys present in first sample
                         if 'engagement_string' in y_sample: label_type_detected = 'engagenet_dict' # Dict containing string
                         elif 'engagement_numeric' in y_sample: label_type_detected = 'daisee_dict' # Dict containing numeric
                         else: label_type_detected = 'unknown_dict'
                     else: # Should not happen if y_batch_sample_collated was dict
                          label_type_detected = 'unknown (error extracting dict)'
                 elif isinstance(y_batch_sample_collated, (list, torch.Tensor)) and len(y_batch_sample_collated) > 0:
                     # Likely a list/tensor of non-dict labels (e.g., strings)
                     y_sample = y_batch_sample_collated[0]
                     if isinstance(y_sample, str): label_type_detected = 'string'
                     else: label_type_detected = f'non-dict ({type(y_sample).__name__})'
                 else: # Handle other cases or empty labels
                     y_sample = y_batch_sample_collated
                     label_type_detected = f'unknown ({type(y_sample).__name__})'

                 results[ds_type]['x_shape'] = str(x_sample.shape)
                 results[ds_type]['label_type'] = label_type_detected
                 print(f"  First valid sample loaded. Detected Label Type: {label_type_detected}, X Shape: {x_sample.shape}")
                 first_sample_loaded = True
            else: print("  Warning: First sample DataLoader yielded no valid batches.")
        except Exception as e: print(f"  Error loading/inspecting first sample: {e}"); traceback.print_exc()

        # 2. Iterate full DataLoader
        # Decide if we need to iterate based on first sample success
        if not first_sample_loaded:
             print("Skipping full dataset iteration as first sample failed.")
        else:
            print(f"Iterating through {ds_type} DataLoader (Batch Size: {inspect_batch_size}) for full inspection...")
            valid_samples_processed, partially_corrupted, fully_corrupted = 0, 0, 0
            dataloader = get_dataloader(config_path, ds_type, batch_size_override=inspect_batch_size, num_workers_override=0)

            if dataloader is None: print(f"  Failed to create main DataLoader for {ds_type}.")
            else:
                is_dict_type = label_type_detected.endswith('_dict') # Check if detected type is dictionary-based

                try:
                    data_iterator = tqdm(dataloader, desc=f"Inspecting {ds_type}", leave=True, unit="batch")
                    for batch in data_iterator:
                        if batch is None: continue
                        x_batch, y_batch_collated = batch; batch_size_actual = x_batch.shape[0]; valid_samples_processed += batch_size_actual

                        # --- Collect ALL label keys if labels are dicts ---
                        if is_dict_type and isinstance(y_batch_collated, dict):
                            results[ds_type]['all_label_keys'].update(y_batch_collated.keys())
                        # --------------------------------------------------

                        # Corruption Check
                        for i in range(batch_size_actual):
                            x_sample = x_batch[i]; num_frames = x_sample.shape[0]; corrupted_frame_count = 0
                            if num_frames == 0: fully_corrupted += 1; continue
                            for frame_idx in range(num_frames):
                                if torch.sum(x_sample[frame_idx]).item() == 0: corrupted_frame_count += 1 # Adjust check if needed
                            if corrupted_frame_count == num_frames: fully_corrupted += 1
                            elif corrupted_frame_count > 0: partially_corrupted += 1

                        # --- Collect Labels for Distribution (simplified) ---
                        # Store raw labels (or relevant part) based on detected type
                        if is_dict_type and isinstance(y_batch_collated, dict):
                             # Store list of dicts for multi-key distribution later
                             keys_in_batch = list(y_batch_collated.keys())
                             num_items = len(y_batch_collated.get(keys_in_batch[0], [])) # Get length from first key
                             for i in range(num_items):
                                 label_dict = {k: y_batch_collated[k][i] for k in keys_in_batch if i < len(y_batch_collated[k])}
                                 all_labels_for_dist.append(label_dict)
                        elif not is_dict_type:
                             # Assume y_batch_collated is a list/tensor of simple labels
                             if isinstance(y_batch_collated, torch.Tensor):
                                 all_labels_for_dist.extend(y_batch_collated.tolist())
                             elif isinstance(y_batch_collated, list):
                                  all_labels_for_dist.extend(y_batch_collated)
                        # ----------------------------------------------------

                    # Store final counts
                    results[ds_type]['valid_sample_count'] = valid_samples_processed
                    results[ds_type]['partially_corrupted_count'] = partially_corrupted
                    results[ds_type]['fully_corrupted_count'] = fully_corrupted

                except Exception as e: print(f"\n  Error during full DataLoader iteration for {ds_type}: {e}"); traceback.print_exc()

                # 3. Calculate Distributions Dynamically
                if all_labels_for_dist:
                    print("Calculating label distributions...")
                    dist_results = {}
                    if is_dict_type:
                        # Iterate through all keys found across the dataset for dict types
                        for key in results[ds_type]['all_label_keys']:
                            # Extract values for this specific key, skipping None
                            key_values = [d.get(key) for d in all_labels_for_dist if isinstance(d, dict) and d.get(key) is not None]
                            if key_values:
                                try:
                                    # Attempt to convert to string for Counter compatibility if mixed types exist
                                    dist_results[key] = Counter(map(str, key_values))
                                except Exception as e:
                                     print(f"Warning: Could not count distribution for key '{key}': {e}")
                    elif not is_dict_type:
                        # Calculate distribution for the single non-dict label type
                        try:
                             dist_results['label'] = Counter(map(str, [lbl for lbl in all_labels_for_dist if lbl is not None]))
                        except Exception as e:
                             print(f"Warning: Could not count distribution for non-dict labels: {e}")

                    results[ds_type]['distributions'] = dist_results
                else:
                    print("  No valid labels collected for distribution.")

        # Finalize collected keys
        if results[ds_type]['all_label_keys']:
             results[ds_type]['all_label_keys'] = sorted(list(results[ds_type]['all_label_keys']))
        else: results[ds_type]['all_label_keys'] = [] # Ensure list


    # --- Final Detailed Summary Print (using helper) ---
    print("\n\n" + "="*30)
    print("--- Detailed Inspection Summary ---")
    print(f"Config File: {config_path}")
    print(f"Results Base Directory (Calculated): {display_base_result_dir}")
    print(f"(Config Name: {display_config_name}, Version: {display_pipeline_version})")
    print("="*30)

    for ds_type, info in results.items():
        print(f"\n{ds_type}:")
        print(f"  Potential Files Found: {info['count']}")
        if info['valid_sample_count'] > 0:
            loaded_count = info['valid_sample_count']
            print(f"  Samples Loaded: {loaded_count}")
            print(f"  X Shape: {info['x_shape']}")
            print(f"  Detected Label Type: {info['label_type']}")
            print(f"  Corruption:")
            print(f"    - Full: {info['fully_corrupted_count']} ({info['fully_corrupted_count']/loaded_count:.1%})")
            print(f"    - Partial: {info['partially_corrupted_count']} ({info['partially_corrupted_count']/loaded_count:.1%})")
            # Use the helper function to print distributions
            print(format_distribution_dict(info['distributions'], indent="  "))
        elif info['count'] == 0: print("  (Empty/Failed Dataset)")
        elif info['count'] > 0: print("  (Dataset Init OK, DataLoader Failed)")
        else: print("  (Directory not found/Scan failed)")
    print("\n" + "-" * 30)


    # --- SIMPLIFIED Snapshot Table ---
    print("\n--- Dataset Snapshot ---")
    if TABULATE_AVAILABLE:
        table_data = []
        for ds_type in dataset_types:
            info = results.get(ds_type)
            row = {"Dataset": ds_type}  # Header 1: Dataset
            dist_str = "(N/A)"  # Default distribution string

            if info and info['valid_sample_count'] > 0:
                # Format the distribution using the helper function for this row
                # Use a smaller indent for the table cell
                dist_str = format_distribution_dict(info.get('distributions'), indent="")

                row["Samples Loaded"] = info['valid_sample_count']  # Header 2
                row["X Shape"] = info.get('x_shape', 'N/A')  # Header 3
                row["Label Distribution"] = dist_str  # Header 4 (Content)

            # Handle cases where loading failed or dataset empty
            elif info and info['count'] == 0:
                row["Samples Loaded"] = 0;
                row["X Shape"] = "N/A";
                row["Label Distribution"] = "(Empty)"
            elif info and info['count'] > 0:
                row["Samples Loaded"] = f"0 (of {info['count']})";
                row["X Shape"] = info.get('x_shape', 'N/A');
                row["Label Distribution"] = "(Load Fail)"
            else:
                row["Samples Loaded"] = "N/A";
                row["X Shape"] = "N/A";
                row["Label Distribution"] = "(Not Found)"
            table_data.append(row)

        print(tabulate(table_data, headers="keys", tablefmt="fancy_grid", stralign="left", numalign="right"))

    else: # Fallback print
        print(" (Install 'tabulate' library for formatted table summary)")
        for ds_type, info in results.items(): print(f" - {ds_type}: {info.get('valid_sample_count','N/A')} samples, Shape: {info.get('x_shape','N/A')}, Type: {info.get('label_type','N/A')}")

    # Print config/version footer separately
    footer = f"Config: {display_config_name} | Version: {display_pipeline_version}"
    print(f"\n{footer}\n" + "-" * (len(footer) + 2))

    return None # Explicit return


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Running inspect function directly for demonstration ---")
    example_config_path = "../Pipelines/configs/ENGAGENET_10fps_quality95_randdist.json"
    example_batch_size = 32
    if CACHE_DIR is None: print("\nCannot run demo: CACHE_DIR unavailable.")
    elif not TABULATE_AVAILABLE: print("\nCannot run demo table: 'tabulate' unavailable.")
    elif os.path.exists(example_config_path): inspect(example_config_path, example_batch_size)
    else: print(f"Example config file not found at: {example_config_path}")
