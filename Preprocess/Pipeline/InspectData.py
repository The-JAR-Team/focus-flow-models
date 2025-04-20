import os
import torch
from torch.utils.data import DataLoader, Dataset # Import Dataset base class
import sys # To potentially add project root to path if needed
import traceback # For detailed error printing
from collections import Counter # For counting distributions
import warnings # To filter warnings
from tqdm import tqdm # Ensure tqdm is imported

# --- Filter specific FutureWarning from torch.load ---
warnings.filterwarnings("ignore", category=FutureWarning, message=".*You are using `torch.load` with `weights_only=False`.*")
# ----------------------------------------------------

# --- Import the Dataset Class ---
try:
    from Preprocess.Pipeline.Encapsulation.CachedTensorDataset import CachedTensorDataset
except ImportError:
    try:
        from cached_tensor_dataset import CachedTensorDataset # If it's in the same folder
    except ImportError:
        print("Error: Could not import CachedTensorDataset.")
        exit()
# --------------------------------

# --- Custom Collate Function ---
def collate_wrapper(batch):
    """ Filters out None samples and manually collates label dicts. """
    filtered_batch = [item for item in batch if item is not None]
    if not filtered_batch: return None
    try:
        tensors = [item[0] for item in filtered_batch]
        labels = [item[1] for item in filtered_batch]
    except (IndexError, TypeError) as e:
         # Use print here as tqdm might not be active when collate fails early
         print(f"Error: Invalid item structure in batch: {e}. Expected (tensor, label). Item: {filtered_batch[0]}")
         return None
    try:
        collated_tensors = torch.stack(tensors, 0)
    except Exception as e:
        print(f"Error stacking tensors: {e}"); return None

    collated_labels = {}
    if labels and isinstance(labels[0], dict):
        all_keys = set().union(*(d.keys() for d in labels))
        for key in all_keys:
            collated_labels[key] = [d.get(key) for d in labels]
    elif labels:
        try:
             from torch.utils.data._utils.collate import default_collate
             collated_labels = default_collate(labels)
        except Exception as e:
             print(f"Error collating non-dict labels: {e}"); collated_labels = labels
    return collated_tensors, collated_labels
# ---------------------------


def inspect(base_result_dir, inspect_batch_size):
    """
    Inspect the dataset in the given directory.
    :param base_result_dir: Base directory containing dataset folders.
    :param inspect_batch_size: Batch size for DataLoader.
    :return: None
    """
    print(f"--- Loading data from base directory: {base_result_dir} ---")

    if not os.path.isdir(base_result_dir):
        print(f"\nError: Base results directory not found at '{base_result_dir}'")
        print(f"Please set BASE_RESULTS_DIR correctly.")
        exit()

    dataset_types = ['Train', 'Validation', 'Test']
    results = {}

    for ds_type in dataset_types:
        # Use standard print for messages before the main tqdm loop
        print(f"\n--- Processing: {ds_type} ---")
        data_dir = os.path.join(base_result_dir, ds_type)
        results[ds_type] = {'count': 0, 'x_shape': None, 'label_type': 'Unknown', 'distributions': {}}
        label_type_detected = None
        all_labels = []

        if not os.path.isdir(data_dir):
            print(f"Directory not found: {data_dir}. Skipping {ds_type}.")
            continue

        # 1. Create Dataset and get count
        try:
            dataset = CachedTensorDataset(data_dir)
            dataset_len = len(dataset)
            print(f"Found {dataset_len} potential samples in {ds_type} dataset.")
            results[ds_type]['count'] = dataset_len
        except Exception as e:
            print(f"Error initializing CachedTensorDataset for {ds_type}: {e}")
            continue

        if dataset_len == 0:
            print("Skipping further inspection for empty dataset.")
            continue

        # 2. Load first sample BEFORE main loop to print info first
        print("Inspecting first valid sample...")
        first_batch_loaded = False
        try:
            # Create a temporary loader just to get the first item
            temp_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_wrapper)
            first_valid_batch = None
            for batch in temp_loader: # Iterate until a non-None batch is found
                 if batch is not None:
                     first_valid_batch = batch
                     break

            if first_valid_batch is not None:
                x_sample = first_valid_batch[0][0]
                y_sample_collated = first_valid_batch[1]
                y_sample = {key: values[0] for key, values in y_sample_collated.items()} if isinstance(y_sample_collated, dict) else y_sample_collated[0]

                results[ds_type]['x_shape'] = str(x_sample.shape)
                if isinstance(y_sample, dict):
                    if y_sample.get('engagement_string') is not None: label_type_detected = 'engagenet'
                    elif y_sample.get('engagement_numeric') is not None: label_type_detected = 'daisee'
                    else: label_type_detected = 'unknown_dict'
                else: label_type_detected = 'unknown_non_dict'
                results[ds_type]['label_type'] = label_type_detected

                print(f"  Detected label type: {label_type_detected}")
                print(f"  First Sample X Shape: {x_sample.shape}")
                first_batch_loaded = True
            else:
                 print("  Could not load a valid first sample.")

        except Exception as e:
             print(f"  Error occurred while inspecting first sample: {e}")
             # traceback.print_exc() # Optionally print full traceback

        # 3. Iterate through full DataLoader for distribution if first sample was loaded
        if first_batch_loaded and label_type_detected not in ['Unknown', 'unknown_dict', 'unknown_non_dict']:
            print(f"Iterating through {ds_type} DataLoader to collect all labels...")
            try:
                # Create the main DataLoader for full iteration
                dataloader = DataLoader(dataset, batch_size=inspect_batch_size, shuffle=False, num_workers=0,
                                        collate_fn=collate_wrapper)
                data_iterator = tqdm(dataloader, desc=f"Collecting {ds_type} labels", leave=True)
                for batch in data_iterator:
                    if batch is None: continue
                    _, y_batch_collated = batch

                    # Collect labels based on the type detected from the first sample
                    if label_type_detected == 'engagenet' and 'engagement_string' in y_batch_collated:
                        all_labels.extend([lbl for lbl in y_batch_collated['engagement_string'] if lbl is not None])
                    elif label_type_detected == 'daisee':
                         num_items_in_batch = len(y_batch_collated.get('engagement_numeric', []))
                         for i in range(num_items_in_batch):
                             label_dict = {
                                 key: y_batch_collated.get(key, [None]*num_items_in_batch)[i]
                                 for key in ['engagement_numeric', 'boredom_numeric', 'confusion_numeric', 'frustration_numeric']
                             }
                             all_labels.append(label_dict)
            except Exception as e:
                 print(f"  Error occurred while iterating full DataLoader for {ds_type}: {e}")
                 traceback.print_exc()

            # 4. Calculate Distributions
            if not all_labels:
                print("  No valid labels collected to calculate distribution.")
            else:
                print("Calculating label distributions...")
                if label_type_detected == 'engagenet':
                    results[ds_type]['distributions']['engagement_string'] = Counter(all_labels)
                elif label_type_detected == 'daisee':
                    for key in ['engagement_numeric', 'boredom_numeric', 'confusion_numeric', 'frustration_numeric']:
                        valid_numeric_labels = [d.get(key) for d in all_labels if d is not None and d.get(key) is not None]
                        if valid_numeric_labels:
                             results[ds_type]['distributions'][key] = Counter(valid_numeric_labels)
        elif not first_batch_loaded:
             print("  Skipping distribution calculation as first sample failed.")
        else:
             print(f"  Skipping distribution calculation due to unknown label type: {label_type_detected}")


    # --- Final Summary ---
    print("\n\n--- Inspection Summary ---")
    print(f"Base Directory: {base_result_dir}")
    for ds_type, info in results.items():
        print(f"\n{ds_type}:")
        print(f"  Total Files Found: {info['count']}")
        if info.get('x_shape') is not None:
            print(f"  Sample X Shape: {info['x_shape']}")
            print(f"  Detected Label Type: {info['label_type']}")
            if info['distributions']:
                print("  Label Distributions:")
                for label_key, counts in info['distributions'].items():
                    sorted_counts = sorted(counts.items())
                    dist_parts = [f"{value}: {count}" for value, count in sorted_counts]
                    print(f"    {label_key}: " + ", ".join(dist_parts))
            else:
                print("  (No distribution calculated)")
        elif info['count'] > 0:
             print("  (Could not load or inspect any valid sample)")
        else:
            print("  (No samples found or directory missing)")
    print("-" * 25)
