# Model/utils.py

import torch
import numpy as np # Import numpy for calculations
from typing import Optional, Tuple, Dict, Any, List

# ================================================
# === Utility Functions ===
# ================================================


# get_targets remains the same
def get_targets(label_batch: Dict[str, List[Any]], label_to_idx_map: Dict[str, int], idx_to_score_map: Dict[int, float]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Converts string labels from a batch dictionary to target score and index tensors.
    (No changes needed here)
    """
    # ... (function implementation remains the same) ...
    if 'engagement_string' not in label_batch:
        return None
    string_labels = label_batch['engagement_string']
    target_scores = []
    target_indices = []
    for lbl in string_labels:
        processed_lbl = str(lbl).strip()
        class_idx = label_to_idx_map.get(processed_lbl)
        if class_idx is None: return None
        score = idx_to_score_map.get(class_idx)
        if score is None: return None
        target_indices.append(class_idx)
        target_scores.append(score)
    scores_tensor = torch.tensor(target_scores, dtype=torch.float).unsqueeze(1)
    indices_tensor = torch.tensor(target_indices, dtype=torch.long)
    return scores_tensor, indices_tensor


# --- **** MODIFIED FUNCTION **** ---
def map_score_to_class_idx(score_tensor: torch.Tensor, idx_to_score_map: Dict[int, float]) -> torch.Tensor:
    """
    Maps continuous regression scores back to discrete class indices (0-4)
    dynamically based on the provided idx_to_score_map.

    It calculates thresholds as midpoints between adjacent class scores.
    Assumes indices 4, 0, 1, 2, 3 represent ascending score ranges.

    Args:
        score_tensor (torch.Tensor): Tensor of predicted scores.
        idx_to_score_map (Dict[int, float]): Mapping from class indices to target scores.

    Returns:
        torch.Tensor: Tensor of predicted class indices.
    """
    # --- Dynamically Calculate Thresholds ---
    # Define the order of indices relevant for boundaries
    boundary_indices = [4, 0, 1, 2, 3] # SNP, Not, Barely, Engaged, Highly

    # Check if all boundary indices are in the map
    if not all(idx in idx_to_score_map for idx in boundary_indices):
        raise ValueError("idx_to_score_map is missing required indices (4, 0, 1, 2, 3) to calculate thresholds.")

    # Get scores in the correct order
    scores = [idx_to_score_map[idx] for idx in boundary_indices]

    # Calculate midpoints between adjacent scores
    thresholds = []
    for i in range(len(scores) - 1):
        midpoint = (scores[i] + scores[i+1]) / 2.0
        thresholds.append(midpoint)
    # thresholds should now be e.g., [mid(s4,s0), mid(s0,s1), mid(s1,s2), mid(s2,s3)]

    # --- Apply Mapping Logic ---
    score_tensor_flat = score_tensor.squeeze()
    # Default to the class index corresponding to the lowest score (index 4)
    preds = torch.full_like(score_tensor_flat, 4, dtype=torch.long)

    # Assign classes based on dynamic thresholds
    # score >= t[0] and score < t[1] maps to index 0 (Not Engaged)
    preds = torch.where((score_tensor_flat >= thresholds[0]) & (score_tensor_flat < thresholds[1]), 0, preds)
    # score >= t[1] and score < t[2] maps to index 1 (Barely Engaged)
    preds = torch.where((score_tensor_flat >= thresholds[1]) & (score_tensor_flat < thresholds[2]), 1, preds)
    # score >= t[2] and score < t[3] maps to index 2 (Engaged)
    preds = torch.where((score_tensor_flat >= thresholds[2]) & (score_tensor_flat < thresholds[3]), 2, preds)
    # score >= t[3] maps to index 3 (Highly Engaged)
    preds = torch.where(score_tensor_flat >= thresholds[3], 3, preds)

    # Restore shape if needed
    if score_tensor.dim() > 1 and score_tensor.shape[-1] == 1 :
        preds = preds.unsqueeze(-1)

    return preds


# --- **** Update __main__ block for testing **** ---
if __name__ == '__main__':
    print("--- Utility Functions Example ---")

    # Example setup (use the latest score map for testing)
    example_label_map = {'Not Engaged': 0, 'Barely Engaged': 1, 'Engaged': 2, 'Highly Engaged': 3, 'SNP': 4}
    # example_score_map_orig = {4: 0.0, 0: 0.25, 1: 0.5, 2: 0.75, 3: 1.0}
    example_score_map_new = {4: 0.05, 0: 0.3, 1: 0.5, 2: 0.7, 3: 0.95}
    current_test_score_map = example_score_map_new # Choose which map to test with

    # Example 1: get_targets
    example_label_batch = {'engagement_string': ['Engaged', 'Not Engaged', 'SNP', 'Highly Engaged']}
    print("\nTesting get_targets:")
    targets_valid = get_targets(example_label_batch, example_label_map, current_test_score_map)
    if targets_valid:
        scores, indices = targets_valid
        print("Valid batch processed:")
        print(f"  Scores Tensor: {scores.T}")
        print(f"  Indices Tensor: {indices}")
    else:
        print("Error processing valid batch.")

    # Example 2: map_score_to_class_idx
    print("\nTesting map_score_to_class_idx (Dynamic Thresholds):")
    print(f"Using Score Map: {current_test_score_map}")
    example_scores = torch.tensor([[0.1], [0.2], [0.35], [0.45], [0.55], [0.65], [0.75], [0.85], [0.98]])
    # Pass the map to the function
    mapped_classes = map_score_to_class_idx(example_scores, current_test_score_map)
    print(f"Input Scores:\n{example_scores}")
    # Expected classes (for new map {4:0.05, 0:0.3, 1:0.5, 2:0.7, 3:0.95}): 4, 0, 0, 1, 1, 2, 2, 3, 3
    print(f"Mapped Classes:\n{mapped_classes}")
