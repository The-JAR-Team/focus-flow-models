import torch
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix as sk_confusion_matrix # Added import
)
from typing import Dict, Any, Tuple, Mapping, Union

# --- (Original map_score_to_class_idx function - no changes needed here, but included for completeness) ---
def map_score_to_class_idx(
        score_tensor: Union[torch.Tensor, np.ndarray],
        idx_to_score_map: Mapping[int, float]
) -> Union[torch.Tensor, np.ndarray, None]:
    """
    Maps continuous regression scores back to discrete class indices.
    Assumes idx_to_score_map provides scores for indices that define ascending score ranges.
    """
    is_torch_input = isinstance(score_tensor, torch.Tensor)
    original_shape = score_tensor.shape

    if is_torch_input:
        device = score_tensor.device
        scores_np = score_tensor.detach().cpu().numpy().squeeze()
    else:
        scores_np = np.array(score_tensor).squeeze()

    if scores_np.ndim == 0: # Handle scalar input
        scores_np = np.array([scores_np])

    if scores_np.size == 0:  # Handle empty array after squeeze
        print("Warning: map_score_to_class_idx received empty score_tensor.")
        return None

    boundary_defining_indices = sorted(idx_to_score_map.keys(), key=lambda k: idx_to_score_map[k])

    if not boundary_defining_indices:
        print("Error: idx_to_score_map must contain at least one entry.")
        return None # Critical error

    if len(boundary_defining_indices) == 1:
        # If only one class score is defined, all predictions map to this class index.
        predicted_indices_np = np.full_like(scores_np, boundary_defining_indices[0], dtype=np.int64)
    else:
        # Create thresholds mid-way between sorted class scores
        class_scores = [idx_to_score_map[idx] for idx in boundary_defining_indices]
        thresholds = [(class_scores[i] + class_scores[i + 1]) / 2.0 for i in range(len(class_scores) - 1)]

        # Initialize all predictions to the first class index (lowest score range)
        predicted_indices_np = np.full_like(scores_np, boundary_defining_indices[0], dtype=np.int64)
        # Iterate through thresholds to assign higher class indices
        for i, threshold in enumerate(thresholds):
            predicted_indices_np[scores_np >= threshold] = boundary_defining_indices[i + 1]

    # Determine output shape, ensuring it's at least 2D like (batch, 1)
    if len(original_shape) > 1 and original_shape[-1] == 1: # e.g. (batch, 1)
        output_shape = original_shape
    elif len(original_shape) == 1 and original_shape[0] > 0:  # (batch,)
        output_shape = (original_shape[0], 1)
    elif scores_np.size > 0: # Fallback for squeezed scalars that became 1D array of size 1
        output_shape = (scores_np.size, 1)
    else: # Should not be reached if empty array handled above
        return None

    if is_torch_input:
        return torch.tensor(predicted_indices_np, dtype=torch.long, device=device).reshape(output_shape)
    else:
        return predicted_indices_np.reshape(output_shape)
# --- (End of map_score_to_class_idx) ---


def compute_metrics(
        eval_pred: Any,  # EvalPrediction object
        idx_to_score_map: Mapping[int, float],
        idx_to_name_map: Mapping[int, str],
        num_classes_classification: int # Used for defining labels in sk_confusion_matrix
) -> Dict[str, Any]: # Return type changed to Any to allow list of lists for CM
    """
    Computes evaluation metrics for the multi-task model.
    Assumes eval_pred.predictions is a tuple: (regression_scores, classification_logits, ...)
    and eval_pred.label_ids is a dict: {'regression_targets': ..., 'classification_targets': ...}

    MODIFIED:
    - Calculates and returns confusion matrices for both logits and mapped scores.
    - Ensures confusion matrix data is JSON serializable (list of lists).
    - Uses `class_labels_for_cm` to ensure consistent CM dimensions.
    """
    raw_predictions_tuple = eval_pred.predictions
    labels_dict = eval_pred.label_ids
    metrics = {}

    # Define class labels for confusion matrix based on idx_to_name_map keys
    # This ensures the CM has consistent dimensions, even if some classes are not predicted/present in a batch.
    # It's important that num_classes_classification aligns with the actual number of classes
    # represented in idx_to_name_map.
    class_labels_for_cm = sorted(list(idx_to_name_map.keys())) if idx_to_name_map else list(range(num_classes_classification))


    # --- Regression Metrics ---
    if isinstance(labels_dict, dict) and 'regression_targets' in labels_dict and \
            isinstance(raw_predictions_tuple, tuple) and len(raw_predictions_tuple) > 0:

        reg_preds_np = np.array(raw_predictions_tuple[0]).squeeze()
        reg_labels_np = np.array(labels_dict['regression_targets']).squeeze()

        if reg_preds_np.ndim == 0: reg_preds_np = np.array([reg_preds_np])
        if reg_labels_np.ndim == 0: reg_labels_np = np.array([reg_labels_np])

        if reg_preds_np.size > 0 and reg_labels_np.size > 0 and reg_preds_np.shape == reg_labels_np.shape:
            try:
                metrics["mse"] = float(mean_squared_error(reg_labels_np, reg_preds_np))
                metrics["mae"] = float(mean_absolute_error(reg_labels_np, reg_preds_np))
                metrics["r2"] = float(r2_score(reg_labels_np, reg_preds_np))
            except ValueError as e:
                print(f"Warning: Could not compute regression metrics. Error: {e}")
                metrics["mse"] = float('nan'); metrics["mae"] = float('nan'); metrics["r2"] = float('nan')
        else:
            print(f"Warning: Regression predictions/labels shape mismatch or empty. Preds shape: {reg_preds_np.shape}, Labels shape: {reg_labels_np.shape}")
            metrics["mse"] = float('nan'); metrics["mae"] = float('nan'); metrics["r2"] = float('nan')
    else:
        print("Warning: Skipping regression metrics. Conditions not met (missing data or unexpected format).")

    # --- Classification Metrics (from classification head's logits) ---
    if isinstance(labels_dict, dict) and 'classification_targets' in labels_dict and \
            isinstance(raw_predictions_tuple, tuple) and len(raw_predictions_tuple) > 1:

        cls_logits_np = np.array(raw_predictions_tuple[1]) # (batch_size, num_classes)
        cls_labels_true_np = np.array(labels_dict['classification_targets']).squeeze() # (batch_size,)

        if cls_logits_np.size > 0 and cls_labels_true_np.size > 0 and cls_logits_np.ndim == 2:
            cls_preds_indices_np = np.argmax(cls_logits_np, axis=-1) # Predicted class indices
            try:
                metrics["cls_accuracy_from_logits"] = float(accuracy_score(cls_labels_true_np, cls_preds_indices_np))
                # Calculate confusion matrix from logits
                cm_logits = sk_confusion_matrix(cls_labels_true_np, cls_preds_indices_np, labels=class_labels_for_cm)
                metrics["confusion_matrix_logits"] = cm_logits.tolist() # Convert to list of lists for JSON
            except ValueError as e:
                print(f"Warning: Could not compute cls_accuracy_from_logits or confusion_matrix_logits. Error: {e}")
                metrics["cls_accuracy_from_logits"] = float('nan')
                metrics["confusion_matrix_logits"] = [([0] * len(class_labels_for_cm)) for _ in class_labels_for_cm] # Empty CM
        else:
            print(f"Warning: Classification logits or true labels are empty/invalid for logits metrics. Logits shape: {cls_logits_np.shape}, Labels shape: {cls_labels_true_np.shape}")
            metrics["cls_accuracy_from_logits"] = float('nan')
            metrics["confusion_matrix_logits"] = [([0] * len(class_labels_for_cm)) for _ in class_labels_for_cm]
    else:
        print("Warning: Skipping classification (logits) metrics. Conditions not met.")

    # --- Mapped Classification Metrics (from regression scores) ---
    if isinstance(labels_dict, dict) and 'classification_targets' in labels_dict and \
            isinstance(raw_predictions_tuple, tuple) and len(raw_predictions_tuple) > 0 and idx_to_score_map:

        reg_preds_for_map_np = np.array(raw_predictions_tuple[0]).squeeze() # Regression scores
        cls_labels_true_for_map_np = np.array(labels_dict['classification_targets']).squeeze() # True class indices

        if reg_preds_for_map_np.size > 0 and cls_labels_true_for_map_np.size > 0:
            # Ensure reg_preds_for_map_np is 2D for map_score_to_class_idx
            if reg_preds_for_map_np.ndim == 0:
                reg_preds_for_map_np_reshaped = np.array([[reg_preds_for_map_np]]) # (1,1)
            elif reg_preds_for_map_np.ndim == 1:
                reg_preds_for_map_np_reshaped = reg_preds_for_map_np.reshape(-1, 1) # (batch_size, 1)
            else: # Assumes it's already (batch_size, 1) or similar
                reg_preds_for_map_np_reshaped = reg_preds_for_map_np

            mapped_cls_preds_indices_np = map_score_to_class_idx(reg_preds_for_map_np_reshaped, idx_to_score_map)

            if mapped_cls_preds_indices_np is not None and mapped_cls_preds_indices_np.size > 0:
                mapped_cls_preds_indices_np = mapped_cls_preds_indices_np.squeeze() # Squeeze to 1D array
                # Ensure shapes match for accuracy_score and confusion_matrix
                if mapped_cls_preds_indices_np.shape == cls_labels_true_for_map_np.shape:
                    try:
                        metrics["cls_accuracy_from_mapped_scores"] = float(
                            accuracy_score(cls_labels_true_for_map_np, mapped_cls_preds_indices_np))
                        # Calculate confusion matrix from mapped scores
                        cm_mapped = sk_confusion_matrix(cls_labels_true_for_map_np, mapped_cls_preds_indices_np, labels=class_labels_for_cm)
                        metrics["confusion_matrix_mapped"] = cm_mapped.tolist() # Convert to list of lists
                    except ValueError as e:
                        print(f"Warning: Could not compute cls_accuracy_from_mapped_scores or confusion_matrix_mapped. Error: {e}")
                        metrics["cls_accuracy_from_mapped_scores"] = float('nan')
                        metrics["confusion_matrix_mapped"] = [([0] * len(class_labels_for_cm)) for _ in class_labels_for_cm]
                else:
                    print(f"Warning: Shape mismatch for mapped classification accuracy. Mapped_preds shape: {mapped_cls_preds_indices_np.shape}, True_labels shape: {cls_labels_true_for_map_np.shape}")
                    metrics["cls_accuracy_from_mapped_scores"] = float('nan')
                    metrics["confusion_matrix_mapped"] = [([0] * len(class_labels_for_cm)) for _ in class_labels_for_cm]
            else:
                print("Warning: map_score_to_class_idx returned None or empty for mapped classification.")
                metrics["cls_accuracy_from_mapped_scores"] = float('nan')
                metrics["confusion_matrix_mapped"] = [([0] * len(class_labels_for_cm)) for _ in class_labels_for_cm]
        else:
            print("Warning: Regression scores or true classification labels for mapping are empty.")
            metrics["cls_accuracy_from_mapped_scores"] = float('nan')
            metrics["confusion_matrix_mapped"] = [([0] * len(class_labels_for_cm)) for _ in class_labels_for_cm]
    else:
        print("Warning: Skipping mapped classification metrics. Conditions not met.")

    # --- Log individual losses if available ---
    if isinstance(raw_predictions_tuple, tuple):
        if len(raw_predictions_tuple) > 2 and raw_predictions_tuple[2] is not None: # eval_loss_regression_component
            try:
                loss_reg_val = np.mean(raw_predictions_tuple[2])
                if not np.isnan(loss_reg_val):
                    metrics["eval_loss_regression_component"] = float(loss_reg_val)
            except Exception as e:
                print(f"Info: Could not process raw_predictions_tuple[2] as regression loss component: {e}")

        if len(raw_predictions_tuple) > 3 and raw_predictions_tuple[3] is not None: # eval_loss_classification_component
            try:
                loss_cls_val = np.mean(raw_predictions_tuple[3])
                if not np.isnan(loss_cls_val):
                    metrics["eval_loss_classification_component"] = float(loss_cls_val)
            except Exception as e:
                print(f"Info: Could not process raw_predictions_tuple[3] as classification loss component: {e}")

    if not metrics:
        print("Warning: Metrics dictionary is empty. This might cause issues with 'metric_for_best_model'.")

    return metrics
