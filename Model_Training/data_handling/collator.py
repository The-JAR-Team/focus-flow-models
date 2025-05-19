import torch
from typing import List, Dict, Tuple, Any, Optional


def multitask_data_collator(
        batch: List[Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]]
) -> Optional[Dict[str, Any]]:
    """
    Collates a batch of (tensor_stack, multi_task_labels_dict) tuples from CachedTensorDataset.

    The model's forward method expects 'x' for the input features and 'labels'
    as a dictionary containing 'regression_targets' and 'classification_targets'.

    Args:
        batch (List[Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]]):
            A list of samples. Each sample is a tuple where the first element
            is the input tensor (X) and the second is a dictionary of label tensors (Y).
            Items can be None if __getitem__ in the Dataset filtered them out.

    Returns:
        Optional[Dict[str, Any]]:
            A dictionary ready to be passed to the model's forward method.
            Expected keys: "x" (batched input tensors) and "labels" (a dictionary
            of batched label tensors: 'regression_targets', 'classification_targets').
            Returns None if the batch is empty after filtering None items or if stacking fails.
    """
    # 1. Filter out None items (e.g., from failed loads in Dataset's __getitem__)
    valid_batch = [item for item in batch if item is not None]

    if not valid_batch:
        # If the batch is empty after filtering, there's nothing to collate.
        # The DataLoader should ideally not pass an entirely empty list if drop_last=False,
        # but it's good to handle.
        return None

    # 2. Separate tensor_stacks (X) and label dictionaries (Y)
    try:
        # Assuming each item in valid_batch is (tensor_stack, label_dict)
        tensor_stacks = [item[0] for item in valid_batch]
        # label_dicts is a list of dictionaries, e.g., [{'regression_targets': tensor, 'classification_targets': tensor}, ...]
        label_dicts = [item[1] for item in valid_batch]
    except (IndexError, TypeError) as e:
        print(f"Error in collator: Batch items do not have expected (tensor, dict) structure. Error: {e}")
        # This might happen if the Dataset __getitem__ returns an unexpected format.
        return None  # Skip this batch

    # 3. Stack the input tensors (X)
    # Assumes all tensor_stacks in a valid batch have the same shape for stacking.
    # The first dimension of the stacked tensor will be the batch size.
    try:
        batched_x = torch.stack(tensor_stacks, 0)
    except Exception as e:
        print(f"Error stacking input tensors ('x') in collator: {e}")
        # This can happen if tensors in the batch have inconsistent shapes.
        # For debugging, you might want to print shapes:
        # for i, t_stack in enumerate(tensor_stacks):
        #     print(f"Collator: tensor_stack {i} shape: {t_stack.shape}")
        return None  # Skip this batch if inputs can't be batched

    # 4. Prepare the batched labels dictionary for the model's forward pass
    # Initialize a dictionary to hold batched labels
    batched_labels_dict = {}

    # Check if label_dicts is not empty and its first element is a dictionary (as expected)
    if label_dicts and isinstance(label_dicts[0], dict):
        # Iterate over the keys found in the first sample's label dictionary
        # (e.g., 'regression_targets', 'classification_targets')
        for key in label_dicts[0].keys():
            try:
                # Collect all tensors for the current key from all samples in the batch
                label_tensors_for_key = [ld[key] for ld in label_dicts]
                # Stack these tensors along the batch dimension
                batched_labels_dict[key] = torch.stack(label_tensors_for_key, 0)
            except KeyError:
                print(
                    f"Warning in collator: Key '{key}' not found in all label dictionaries in the batch. Skipping this key for labels.")
            except Exception as e:
                print(f"Error stacking label tensors for key '{key}' in collator: {e}")
                # If a specific label type can't be batched, we might choose to exclude it
                # or fail the whole batch. For now, let's try to continue if other keys are fine,
                # but this indicates a problem in the data or dataset.
                # If a required label (like 'regression_targets') fails, the model will error.
                pass  # Or return None if any label key fails to stack
    else:
        # This case should ideally not be reached if CachedTensorDataset always returns a label dict.
        print("Warning in collator: label_dicts is empty or not in the expected format (list of dicts).")
        # Depending on model requirements, you might return None or an empty labels dict.
        # If labels are essential for the model, returning None is safer.
        return None

    # 5. Construct the final dictionary to be passed to the model
    # The model's forward method will receive these as arguments.
    # For Hugging Face Trainer, it typically expects inputs to the model
    # and a 'labels' key if loss is to be computed by Trainer/model.
    # Our model computes loss internally if 'labels' (this dict) is passed.
    final_batch = {
        "x": batched_x,  # This will be passed as the 'x' argument to model.forward
        "labels": batched_labels_dict  # This will be passed as the 'labels' argument
    }

    return final_batch