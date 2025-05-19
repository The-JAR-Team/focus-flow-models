# engagement_hf_trainer/utils/metrics.py

import torch
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    classification_report as sk_classification_report,
    confusion_matrix as sk_confusion_matrix
)
from typing import Dict, Any, Tuple, Mapping, Union  # Added Union

def map_score_to_class_idx(
        score_tensor: Union[torch.Tensor, np.ndarray],
        idx_to_score_map: Mapping[int, float]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Maps continuous regression scores back to discrete class indices.
    Assumes idx_to_score_map provides scores for indices that define ascending score ranges.

    Args:
        score_tensor (Union[torch.Tensor, np.ndarray]): Tensor/array of predicted scores.
        idx_to_score_map (Mapping[int, float]): Mapping from class indices to their representative scores.
                                                e.g., {4: 0.05, 0: 0.3, 1: 0.5, 2: 0.7, 3: 0.95}

    Returns:
        Union[torch.Tensor, np.ndarray]: Tensor/array of predicted class indices.
    """
    is_torch_input = isinstance(score_tensor, torch.Tensor)
    original_shape = score_tensor.shape  # Store original shape to restore at the end

    if is_torch_input:
        device = score_tensor.device
        # Squeeze to handle both (batch_size,) and (batch_size, 1)
        scores_np = score_tensor.detach().cpu().numpy().squeeze()
    else:
        scores_np = np.array(score_tensor).squeeze()

    if scores_np.ndim == 0:  # Handle scalar input
        scores_np = np.array([scores_np])

    # Sort the class indices based on their scores to define ordered bins
    boundary_defining_indices = sorted(idx_to_score_map.keys(), key=lambda k: idx_to_score_map[k])

    if len(boundary_defining_indices) < 1:  # Need at least one class
        raise ValueError("idx_to_score_map must contain at least one entry.")
    if len(boundary_defining_indices) == 1:  # Only one class, all scores map to it
        predicted_indices_np = np.full_like(scores_np, boundary_defining_indices[0], dtype=np.int64)
    else:
        class_scores = [idx_to_score_map[idx] for idx in boundary_defining_indices]
        thresholds = []
        for i in range(len(class_scores) - 1):
            midpoint = (class_scores[i] + class_scores[i + 1]) / 2.0
            thresholds.append(midpoint)

        # Initialize predictions to the class with the lowest score range
        predicted_indices_np = np.full_like(scores_np, boundary_defining_indices[0], dtype=np.int64)

        # Iterate through thresholds to assign classes
        # A score x gets class C_j if T_{j-1} <= x < T_j
        # Scores >= last_threshold get the last class
        for i in range(len(thresholds)):
            # Scores >= current threshold get the next class in sorted order
            predicted_indices_np[scores_np >= thresholds[i]] = boundary_defining_indices[i + 1]

    # Reshape to match original input's batch dimension, ensuring output is (batch_size, 1) or similar
    if len(original_shape) > 1 and original_shape[-1] == 1:  # e.g. (batch, 1)
        output_shape = (original_shape[0], 1)
    elif len(original_shape) == 1:  # e.g. (batch,)
        output_shape = (original_shape[0], 1)  # Convert to (batch,1) for consistency
    else:  # Fallback, should ideally match input's structure or a defined standard
        output_shape = (len(scores_np), 1) if scores_np.ndim > 0 else (1, 1)

    if is_torch_input:
        return torch.tensor(predicted_indices_np, dtype=torch.long, device=device).reshape(output_shape)
    else:
        return predicted_indices_np.reshape(output_shape)


def compute_metrics(
        eval_pred,
        idx_to_score_map: Mapping[int, float],
        idx_to_name_map: Mapping[int, str],  # For potential classification report
        num_classes_classification: int  # For potential classification report
) -> Dict[str, float]:
    """
    Computes evaluation metrics for the multi-task model.
    Configurations like idx_to_score_map are passed as arguments.

    Args:
        eval_pred: An EvalPrediction object from Hugging Face Trainer.
                   eval_pred.predictions is the output dict from the model (excluding loss).
                   eval_pred.label_ids is the 'labels' dict from the collator.
        idx_to_score_map (Mapping[int, float]): Mapping from class indices to target scores.
        idx_to_name_map (Mapping[int, str]): Mapping from class indices to class names.
        num_classes_classification (int): Total number of unique classes for classification.

    Returns:
        Dict[str, float]: A dictionary of computed metrics.
    """
    predictions_dict = eval_pred.predictions
    labels_dict = eval_pred.label_ids

    metrics = {}

    # --- Regression Metrics ---
    if 'regression_scores' in predictions_dict and 'regression_targets' in labels_dict:
        reg_preds_np = np.array(predictions_dict['regression_scores']).squeeze()
        reg_labels_np = np.array(labels_dict['regression_targets']).squeeze()

        if reg_preds_np.ndim == 0: reg_preds_np = np.array([reg_preds_np])
        if reg_labels_np.ndim == 0: reg_labels_np = np.array([reg_labels_np])

        if reg_preds_np.size > 0 and reg_labels_np.size > 0 and reg_preds_np.shape == reg_labels_np.shape:
            try:
                metrics["mse"] = mean_squared_error(reg_labels_np, reg_preds_np)
                metrics["mae"] = mean_absolute_error(reg_labels_np, reg_preds_np)
                metrics["r2"] = r2_score(reg_labels_np, reg_preds_np)
            except ValueError as e:
                print(f"Warning: Could not compute regression metrics. Error: {e}")
                metrics["mse"] = float('nan')
                metrics["mae"] = float('nan')
                metrics["r2"] = float('nan')
        else:
            print(
                f"Warning: Regression predictions/labels shape mismatch or empty. Preds shape: {reg_preds_np.shape}, Labels shape: {reg_labels_np.shape}")
            metrics["mse"] = float('nan')
            metrics["mae"] = float('nan')
            metrics["r2"] = float('nan')

    # --- Classification Metrics (from classification head's logits) ---
    if 'classification_logits' in predictions_dict and 'classification_targets' in labels_dict:
        cls_logits_np = np.array(predictions_dict['classification_logits'])
        cls_labels_true_np = np.array(labels_dict['classification_targets']).squeeze()

        if cls_logits_np.size > 0 and cls_labels_true_np.size > 0:
            cls_preds_indices_np = np.argmax(cls_logits_np, axis=-1)
            try:
                metrics["cls_accuracy_from_logits"] = accuracy_score(cls_labels_true_np, cls_preds_indices_np)
            except ValueError as e:
                print(f"Warning: Could not compute cls_accuracy_from_logits. Error: {e}")
                metrics["cls_accuracy_from_logits"] = float('nan')
        else:
            print("Warning: Classification logits or true labels are empty.")
            metrics["cls_accuracy_from_logits"] = float('nan')

    # --- Mapped Classification Metrics (from regression scores) ---
    if 'regression_scores' in predictions_dict and 'classification_targets' in labels_dict and idx_to_score_map:
        reg_preds_for_map_np = np.array(predictions_dict['regression_scores']).squeeze()
        cls_labels_true_for_map_np = np.array(labels_dict['classification_targets']).squeeze()

        if reg_preds_for_map_np.size > 0 and cls_labels_true_for_map_np.size > 0:
            # map_score_to_class_idx expects 1D or 2D (batch, 1) scores
            # Ensure reg_preds_for_map_np is correctly shaped before passing
            if reg_preds_for_map_np.ndim == 0:  # if scalar
                reg_preds_for_map_np_reshaped = np.array([reg_preds_for_map_np])
            elif reg_preds_for_map_np.ndim == 1:  # (batch_size,)
                reg_preds_for_map_np_reshaped = reg_preds_for_map_np.reshape(-1, 1)  # Make it (batch_size, 1)
            else:  # Already (batch_size, 1) or more, use as is
                reg_preds_for_map_np_reshaped = reg_preds_for_map_np

            mapped_cls_preds_indices_np = map_score_to_class_idx(reg_preds_for_map_np_reshaped, idx_to_score_map)
            mapped_cls_preds_indices_np = mapped_cls_preds_indices_np.squeeze()  # Ensure 1D for accuracy_score

            try:
                metrics["cls_accuracy_from_mapped_scores"] = accuracy_score(cls_labels_true_for_map_np,
                                                                            mapped_cls_preds_indices_np)
            except ValueError as e:
                print(f"Warning: Could not compute cls_accuracy_from_mapped_scores. Error: {e}")
                metrics["cls_accuracy_from_mapped_scores"] = float('nan')
        else:
            print("Warning: Regression scores or true classification labels for mapping are empty.")
            metrics["cls_accuracy_from_mapped_scores"] = float('nan')

    # Optional: Detailed classification report using idx_to_name_map and num_classes_classification
    # This can be added here if desired, similar to the commented-out section in the previous version.
    # Example:
    # if 'classification_logits' in predictions_dict and 'classification_targets' in labels_dict:
    #     target_names = [idx_to_name_map.get(i, f"Class_{i}") for i in range(num_classes_classification)]
    #     # ... (generate sk_classification_report) ...

    return metrics
