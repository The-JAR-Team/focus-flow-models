import torch
from typing import Optional, Tuple, Dict, Any, List


# ================================================
# === Utility Functions ===
# ================================================
def get_targets(label_batch: Dict[str, List[Any]], label_to_idx_map: Dict[str, int], idx_to_score_map: Dict[int, float]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Converts string labels from a batch dictionary to target score and index tensors.

    Args:
        label_batch (Dict[str, List[Any]]): Dictionary containing label information,
                                            expected to have 'engagement_string' key.
        label_to_idx_map (Dict[str, int]): Mapping from string labels to class indices.
        idx_to_score_map (Dict[int, float]): Mapping from class indices to regression scores.

    Returns:
        Optional[Tuple[torch.Tensor, torch.Tensor]]: A tuple containing:
            - scores_tensor (torch.Tensor): Target scores (float, shape [batch_size, 1]).
            - indices_tensor (torch.Tensor): Target class indices (long, shape [batch_size]).
        Returns None if 'engagement_string' is missing or any label is invalid/unknown.
    """
    # Check if the required key exists in the batch
    if 'engagement_string' not in label_batch:
        # print("Warning: 'engagement_string' key missing in label batch.") # Optional warning
        return None # Skip batch if key is missing

    string_labels = label_batch['engagement_string']
    target_scores = []
    target_indices = []

    # Process each label in the batch
    for lbl in string_labels:
        processed_lbl = str(lbl).strip() # Handle potential non-string types and whitespace
        class_idx = label_to_idx_map.get(processed_lbl)

        # Check if the label is known
        if class_idx is None:
            # print(f"Warning: Unknown label encountered: '{lbl}'. Skipping batch.") # Optional warning
            return None # Skip entire batch if one label is unknown

        # Get the corresponding score
        score = idx_to_score_map.get(class_idx)
        if score is None:
            # This check ensures map consistency, should ideally not happen if maps are correct
            # print(f"Warning: No score defined for index {class_idx} ('{processed_lbl}'). Skipping batch.") # Optional warning
            return None # Skip batch if score mapping fails

        target_indices.append(class_idx)
        target_scores.append(score)

    # Convert lists to tensors
    # Ensure scores are float and have shape (batch_size, 1) for MSELoss compatibility
    scores_tensor = torch.tensor(target_scores, dtype=torch.float).unsqueeze(1)
    # Indices are long integers, suitable for classification metrics or loss functions
    indices_tensor = torch.tensor(target_indices, dtype=torch.long)

    return scores_tensor, indices_tensor


def map_score_to_class_idx(score_tensor: torch.Tensor) -> torch.Tensor:
    """
    Maps continuous regression scores [0, 1] back to discrete class indices (0-4).

    Mapping Logic:
        SNP(4):    [0.0,   0.125) -> Mapped from score 0.0
        Not(0):    [0.125, 0.375) -> Mapped from score 0.25
        Barely(1): [0.375, 0.625) -> Mapped from score 0.5
        Engaged(2):[0.625, 0.875) -> Mapped from score 0.75
        Highly(3): [0.875, 1.0]   -> Mapped from score 1.0

    Args:
        score_tensor (torch.Tensor): Tensor of predicted scores, expected shape [batch_size, 1] or [batch_size].

    Returns:
        torch.Tensor: Tensor of predicted class indices (long, shape matching input after squeeze).
    """
    # Define the thresholds that separate the classes
    thresholds = [0.125, 0.375, 0.625, 0.875]

    # Ensure tensor is flat for thresholding logic, handles shapes like [batch, 1] or [batch]
    score_tensor_flat = score_tensor.squeeze()

    # Initialize predictions tensor. Default to class 4 (SNP)
    # Using torch.full_like preserves device and basic type (float -> float, int -> int)
    # We specify dtype=torch.long explicitly for class indices.
    preds = torch.full_like(score_tensor_flat, 4, dtype=torch.long)

    # Assign classes based on thresholds using torch.where for efficiency
    # torch.where(condition, value_if_true, value_if_false)
    # Note: Conditions are mutually exclusive except for the >= thresholds[3] case
    preds = torch.where((score_tensor_flat >= thresholds[0]) & (score_tensor_flat < thresholds[1]), 0, preds) # Not Engaged
    preds = torch.where((score_tensor_flat >= thresholds[1]) & (score_tensor_flat < thresholds[2]), 1, preds) # Barely Engaged
    preds = torch.where((score_tensor_flat >= thresholds[2]) & (score_tensor_flat < thresholds[3]), 2, preds) # Engaged
    preds = torch.where(score_tensor_flat >= thresholds[3], 3, preds) # Highly Engaged

    # Restore original shape's last dimension if input was (batch, 1)
    # This ensures output shape matches target_indices shape if needed
    if score_tensor.dim() > 1 and score_tensor.shape[-1] == 1 :
        preds = preds.unsqueeze(-1)

    return preds


if __name__ == '__main__':
    # Example usage of the utility functions
    print("--- Utility Functions Example ---")

    # Example 1: get_targets
    example_label_batch = {'engagement_string': ['Engaged', 'Not Engaged', 'SNP', 'invalid']}
    example_label_map = {'Not Engaged': 0, 'Barely Engaged': 1, 'Engaged': 2, 'Highly Engaged': 3, 'SNP': 4}
    example_score_map = {4: 0.0, 0: 0.25, 1: 0.5, 2: 0.75, 3: 1.0}

    print("\nTesting get_targets:")
    targets = get_targets(example_label_batch, example_label_map, example_score_map)
    if targets:
        print("Should fail due to 'invalid' label.") # Expected behavior
    else:
        print("Correctly returned None for invalid batch.")

    example_label_batch_valid = {'engagement_string': ['Engaged', 'Not Engaged', 'SNP', 'Highly Engaged']}
    targets_valid = get_targets(example_label_batch_valid, example_label_map, example_score_map)
    if targets_valid:
        scores, indices = targets_valid
        print("Valid batch processed:")
        print(f"  Scores Tensor: {scores.T}") # Transpose for better printing
        print(f"  Indices Tensor: {indices}")
    else:
        print("Error processing valid batch.")

    # Example 2: map_score_to_class_idx
    print("\nTesting map_score_to_class_idx:")
    example_scores = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9], [0.05]])
    mapped_classes = map_score_to_class_idx(example_scores)
    print(f"Input Scores:\n{example_scores}")
    print(f"Mapped Classes:\n{mapped_classes}")

    example_scores_flat = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9, 0.05])
    mapped_classes_flat = map_score_to_class_idx(example_scores_flat)
    print(f"\nInput Scores (flat): {example_scores_flat}")
    print(f"Mapped Classes (flat): {mapped_classes_flat}")
