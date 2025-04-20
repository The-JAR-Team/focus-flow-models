import os
import torch
from torch.utils.data import DataLoader, Dataset # Import Dataset base class
# import sys # Not currently used
import traceback # For detailed error printing
from collections import Counter # For counting distributions
import warnings # To filter warnings
from tqdm import tqdm # Ensure tqdm is imported

# --- Filter specific FutureWarning from torch.load ---
# Suppresses warnings about loading tensors without specifying weights_only, common when loading arbitrary data.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*You are using `torch.load` with `weights_only=False`.*")
# ----------------------------------------------------

# --- Import the Dataset Class ---
# Assuming CachedTensorDataset is correctly located and importable
try:
    from Preprocess.Pipeline.Encapsulation.CachedTensorDataset import CachedTensorDataset
except ImportError as e:
    print(f"Error: Could not import CachedTensorDataset: {e}")
    print("Please ensure CachedTensorDataset is accessible.")
    exit()
# --------------------------------

# --- Custom Collate Function ---
def collate_wrapper(batch):
    """ Filters out None samples and manually collates label dicts if present. """
    # Step 1: Filter out any samples that failed to load (returned as None by Dataset's __getitem__)
    filtered_batch = [item for item in batch if item is not None]
    if not filtered_batch:
        # print("Warning: collate_wrapper received an entirely None batch.") # Can be noisy
        return None # Return None if the whole batch was invalid

    # Step 2: Separate tensors and labels
    try:
        # Assuming each valid item is a tuple (tensor, label)
        tensors = [item[0] for item in filtered_batch]
        labels = [item[1] for item in filtered_batch]
    except (IndexError, TypeError) as e:
         # Error if items are not tuples or don't have at least two elements
         print(f"Error: Invalid item structure in batch: {e}. Expected (tensor, label). First item: {filtered_batch[0]}")
         return None # Indicate batch failure

    # Step 3: Collate Tensors (usually stacking)
    try:
        # Stack tensors along a new batch dimension (dimension 0)
        collated_tensors = torch.stack(tensors, 0)
    except Exception as e:
        # Catch errors during stacking (e.g., tensors have inconsistent shapes)
        print(f"Error stacking tensors: {e}")
        # Optionally print shapes for debugging:
        # for i, t in enumerate(tensors): print(f" Tensor {i} shape: {t.shape}")
        return None # Indicate batch failure

    # Step 4: Collate Labels
    collated_labels = {}
    if labels: # Check if there are any labels to process
        first_label = labels[0]
        if isinstance(first_label, dict):
            # Handle dictionary labels (like DAiSEE)
            all_keys = set().union(*(d.keys() for d in labels if isinstance(d, dict))) # Get all unique keys across label dicts
            for key in all_keys:
                # Create a list for each key, extracting value or None if key missing
                collated_labels[key] = [d.get(key) for d in labels if isinstance(d, dict)]
        else:
            # Handle non-dictionary labels (like EngageNet strings or simple numerics)
            try:
                # Use PyTorch's default collate for simpler types if possible
                from torch.utils.data._utils.collate import default_collate
                collated_labels = default_collate(labels)
            except Exception as e:
                # Fallback if default_collate fails
                print(f"Warning: Error using default_collate for labels: {e}. Returning labels as a list.")
                collated_labels = labels # Return as a simple list
    # else: collated_labels remains {}

    return collated_tensors, collated_labels
# ---------------------------


def inspect(base_result_dir, inspect_batch_size):
    """
    Inspect the dataset in the given directory, including corruption checks.
    :param base_result_dir: Base directory containing dataset type folders (Train, Validation, Test).
    :param inspect_batch_size: Batch size for DataLoader during inspection.
    :return: None
    """
    print(f"--- Loading data from base directory: {base_result_dir} ---")

    if not os.path.isdir(base_result_dir):
        print(f"\nError: Base results directory not found at '{base_result_dir}'")
        print(f"Please ensure BASE_RESULTS_DIR points to the correct location containing Train/Validation/Test subfolders.")
        exit()

    dataset_types = ['Train', 'Validation', 'Test']
    results = {} # Dictionary to store inspection results for each dataset type

    for ds_type in dataset_types:
        print(f"\n--- Processing: {ds_type} ---")
        data_dir = os.path.join(base_result_dir, ds_type)

        # Initialize results structure for this dataset type
        results[ds_type] = {
            'count': 0,
            'valid_sample_count': 0, # Count samples successfully loaded by DataLoader
            'x_shape': None,
            'label_type': 'Unknown',
            'distributions': {},
            'partially_corrupted_count': 0, # Samples with *some* bad frames
            'fully_corrupted_count': 0      # Samples where *all* frames are bad
        }
        label_type_detected = None
        all_labels_for_dist = [] # List to collect labels for distribution calculation


        if not os.path.isdir(data_dir):
            print(f"Directory not found: {data_dir}. Skipping {ds_type}.")
            continue

        # 1. Create Dataset and get initial file count
        try:
            # CachedTensorDataset likely finds all .pt files recursively
            dataset = CachedTensorDataset(data_dir)
            dataset_len = len(dataset)
            print(f"Found {dataset_len} potential sample files in {ds_type} directory.")
            results[ds_type]['count'] = dataset_len
        except Exception as e:
            print(f"Error initializing CachedTensorDataset for {ds_type}: {e}")
            traceback.print_exc() # Show details on dataset init error
            continue # Skip to next dataset type

        if dataset_len == 0:
            print("Skipping further inspection for empty dataset directory.")
            continue

        # 2. Attempt to load and inspect the first valid sample using DataLoader
        # This helps detect basic loading issues and sample structure early.
        print("Attempting to load first valid sample...")
        first_sample_loaded = False
        try:
            # Use batch_size=1 and num_workers=0 for simplicity in loading just one item
            # collate_fn handles potential None returns from dataset's __getitem__
            temp_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_wrapper)
            first_valid_batch = next(iter(temp_loader), None) # Get first batch, or None if loader is empty

            if first_valid_batch is not None:
                # Successfully loaded a batch (even if it contains only one item)
                x_batch_sample, y_batch_sample_collated = first_valid_batch
                x_sample = x_batch_sample[0] # Get the first tensor from the batch
                y_sample = None # Initialize y_sample

                # Extract the first label from the potentially collated structure
                if isinstance(y_batch_sample_collated, dict):
                    # Handle dictionary labels (e.g., DAiSEE)
                    y_sample = {key: values[0] for key, values in y_batch_sample_collated.items() if values} # Get first item from each list
                elif isinstance(y_batch_sample_collated, (list, torch.Tensor)):
                    # Handle list or tensor labels (e.g., EngageNet string)
                    if len(y_batch_sample_collated) > 0:
                       y_sample = y_batch_sample_collated[0]
                else:
                    # Handle other potential label types if necessary
                     y_sample = y_batch_sample_collated # Assume it's a single item if not dict/list/tensor

                results[ds_type]['x_shape'] = str(x_sample.shape) # Store shape as string

                # Detect label type based on content
                if isinstance(y_sample, dict):
                    # Simple checks based on expected keys
                    if y_sample.get('engagement_string') is not None: label_type_detected = 'engagenet'
                    elif y_sample.get('engagement_numeric') is not None: label_type_detected = 'daisee'
                    else: label_type_detected = 'unknown_dict'
                elif isinstance(y_sample, str): # Check if it looks like EngageNet string label
                     label_type_detected = 'engagenet_string' # Be more specific if needed
                else: label_type_detected = f'unknown ({type(y_sample).__name__})' # Report type if unknown

                results[ds_type]['label_type'] = label_type_detected

                print(f"  First valid sample loaded.")
                print(f"  Detected label type: {label_type_detected}")
                print(f"  Sample X Shape: {x_sample.shape}")
                # print(f"  Sample Y: {y_sample}") # Optionally print first label
                first_sample_loaded = True
            else:
                 print("  Warning: DataLoader did not yield any valid batches (check collate_fn and dataset).")

        except Exception as e:
             print(f"  Error occurred while loading/inspecting first sample: {e}")
             traceback.print_exc() # Print full traceback for debugging initial load issues

        # 3. Iterate through full DataLoader for distributions and corruption checks
        # Only proceed if the first sample inspection was successful and label type is known enough
        can_calculate_dist = first_sample_loaded and label_type_detected not in ['Unknown', 'unknown_dict', 'unknown_non_dict', 'unknown (NoneType)', 'unknown (type)']

        if not first_sample_loaded:
            print("Skipping full dataset iteration as first sample failed to load.")
        elif not can_calculate_dist:
            print(f"Skipping full dataset iteration due to unknown/unhandled label type: {label_type_detected}")
        else:
            print(f"Iterating through {ds_type} DataLoader (Batch Size: {inspect_batch_size}) for full inspection...")
            valid_samples_processed = 0
            partially_corrupted = 0
            fully_corrupted = 0
            try:
                # Create the main DataLoader for full iteration
                dataloader = DataLoader(dataset, batch_size=inspect_batch_size, shuffle=False, num_workers=0, # Use 0 workers for deterministic inspection
                                        collate_fn=collate_wrapper) # Use the robust collate function
                data_iterator = tqdm(dataloader, desc=f"Inspecting {ds_type}", leave=True, unit="batch")

                for batch in data_iterator:
                    if batch is None:
                        # This might happen if collate_wrapper returns None for a whole batch
                        # print("Warning: Encountered a None batch during iteration.")
                        continue # Skip this batch

                    x_batch, y_batch_collated = batch
                    batch_size_actual = x_batch.shape[0] # Actual number of samples in this batch
                    valid_samples_processed += batch_size_actual

                    # --- Corruption Check ---
                    for i in range(batch_size_actual):
                        x_sample = x_batch[i]
                        num_frames = x_sample.shape[0]
                        corrupted_frame_count = 0
                        if num_frames == 0: # Handle case of tensor with 0 frames
                             fully_corrupted += 1 # Count as fully corrupted if no frames
                             continue

                        for frame_idx in range(num_frames):
                            frame_features = x_sample[frame_idx]
                            # --- !!! ASSUMPTION: Corrupted frame features sum to 0 !!! ---
                            # --- !!! Adjust this check based on your actual placeholder !!! ---
                            # Example checks:
                            # if torch.sum(frame_features).item() == 0:
                            # if torch.isnan(frame_features).any():
                            # if torch.all(frame_features == -999): # If using a specific value
                            if torch.sum(frame_features).item() == 0: # Using sum=0 as placeholder check
                                corrupted_frame_count += 1

                        # Classify sample corruption based on count
                        if corrupted_frame_count == num_frames:
                            fully_corrupted += 1
                        elif corrupted_frame_count > 0:
                            partially_corrupted += 1
                    # -------------------------

                    # --- Collect Labels for Distribution ---
                    if label_type_detected == 'engagenet' or label_type_detected == 'engagenet_string':
                         if 'engagement_string' in y_batch_collated:
                             all_labels_for_dist.extend([lbl for lbl in y_batch_collated['engagement_string'] if lbl is not None])
                         elif isinstance(y_batch_collated, list): # Handle direct string list if collate changed
                              all_labels_for_dist.extend([lbl for lbl in y_batch_collated if isinstance(lbl, str)])
                    elif label_type_detected == 'daisee':
                         num_items_in_batch = x_batch.shape[0] # Use actual batch size
                         keys_to_extract = ['engagement_numeric', 'boredom_numeric', 'confusion_numeric', 'frustration_numeric']
                         # Ensure all expected keys exist in the collated dict, default to list of Nones if not
                         label_data = {key: y_batch_collated.get(key, [None] * num_items_in_batch) for key in keys_to_extract}

                         for i in range(num_items_in_batch):
                             # Construct individual label dict for this sample
                             sample_label_dict = {key: label_data[key][i] for key in keys_to_extract}
                             all_labels_for_dist.append(sample_label_dict)
                    # ---------------------------------------

                # Store results after iterating through all batches
                results[ds_type]['valid_sample_count'] = valid_samples_processed
                results[ds_type]['partially_corrupted_count'] = partially_corrupted
                results[ds_type]['fully_corrupted_count'] = fully_corrupted

            except Exception as e:
                 print(f"\n  Error occurred during full DataLoader iteration for {ds_type}: {e}")
                 traceback.print_exc()

            # 4. Calculate Distributions if labels were collected
            if not all_labels_for_dist:
                print("  No valid labels collected to calculate distribution.")
            else:
                print("Calculating label distributions...")
                dist_results = {}
                if label_type_detected == 'engagenet' or label_type_detected == 'engagenet_string':
                    # Ensure labels are strings before counting
                    string_labels = [str(lbl) for lbl in all_labels_for_dist if lbl is not None]
                    if string_labels:
                        dist_results['engagement_string'] = Counter(string_labels)
                elif label_type_detected == 'daisee':
                    # Calculate distribution for each DAiSEE numeric key
                    for key in ['engagement_numeric', 'boredom_numeric', 'confusion_numeric', 'frustration_numeric']:
                        # Extract valid numeric values for the key
                        valid_numeric_labels = [d.get(key) for d in all_labels_for_dist if isinstance(d, dict) and d.get(key) is not None]
                        if valid_numeric_labels:
                             # Count occurrences of each numeric value
                             dist_results[key] = Counter(valid_numeric_labels)

                results[ds_type]['distributions'] = dist_results # Store calculated distributions


    # --- Final Summary ---
    print("\n\n" + "="*30)
    print("--- Inspection Summary ---")
    print(f"Base Directory: {base_result_dir}")
    print("="*30)
    for ds_type, info in results.items():
        print(f"\n{ds_type}:")
        print(f"  Potential Files Found: {info['count']}")
        if info['valid_sample_count'] > 0: # Check if any samples were actually loaded
            print(f"  Samples Successfully Loaded: {info['valid_sample_count']}")
            print(f"  Sample X Shape: {info['x_shape']}")
            print(f"  Detected Label Type: {info['label_type']}")
            print(f"  Corruption Summary:")
            print(f"    - Fully Corrupted (all frames invalid): {info['fully_corrupted_count']} ({info['fully_corrupted_count']/info['valid_sample_count']:.1%} of loaded)")
            print(f"    - Partially Corrupted (some frames invalid): {info['partially_corrupted_count']} ({info['partially_corrupted_count']/info['valid_sample_count']:.1%} of loaded)")
            if info['distributions']:
                print("  Label Distributions:")
                for label_key, counts in info['distributions'].items():
                    # Sort counts by label value for consistent output
                    try:
                        # Attempt numeric sort first
                        sorted_counts = sorted(counts.items(), key=lambda item: float(item[0]))
                    except ValueError:
                        # Fallback to string sort if conversion fails
                        sorted_counts = sorted(counts.items())
                    # Format distribution string
                    dist_parts = [f"{value}: {count}" for value, count in sorted_counts]
                    print(f"    {label_key}: " + ", ".join(dist_parts))
            else:
                print("  (No distribution calculated or collected)")
        elif info['count'] > 0:
             print("  (Could not load or inspect any valid samples using DataLoader)")
        else:
            print("  (Directory empty or not found)")
    print("\n" + "-" * 30)