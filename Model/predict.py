# Model/predict.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple
# --- **** Import the modified utils function **** ---
from Model.utils import map_score_to_class_idx

# ================================================
# === Prediction Function ===
# ================================================

# --- **** MODIFIED SIGNATURE **** ---
def predict_engagement(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    idx_to_name_map: Dict[int, str],
    idx_to_score_map: Dict[int, float] # Add score map argument
    ) -> Tuple[List[float], List[str]]:
    """
    Predicts engagement scores and mapped class names for a given data loader.

    Args:
        model (nn.Module): The trained model instance.
        data_loader (DataLoader): DataLoader containing the input data.
        device (torch.device): The device to run prediction on.
        idx_to_name_map (Dict[int, str]): Mapping from class indices to class names.
        idx_to_score_map (Dict[int, float]): Mapping from class indices to target scores
                                            (used for dynamic thresholding).

    Returns:
        Tuple[List[float], List[str]]: Predicted scores and class names.
    """
    model.eval()
    pred_scores = []
    pred_classes_names = [] # Renamed for clarity
    print("\n--- Making Predictions ---")

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Predicting", leave=False, ncols=100)
        for batch in pbar:
             if isinstance(batch, (list, tuple)): inputs = batch[0]
             else: inputs = batch
             if not isinstance(inputs, torch.Tensor): continue

             inputs = inputs.to(device)
             outputs = model(inputs)
             scores = outputs.squeeze(-1)

             # Map scores to class indices using dynamic thresholds
             # --- **** MODIFIED CALL **** ---
             classes_idx = map_score_to_class_idx(scores, idx_to_score_map)

             # Store results
             pred_scores.extend(scores.cpu().numpy().tolist())
             # Map indices to names
             pred_classes_names.extend([idx_to_name_map.get(c.item(), "Unknown") for c in classes_idx])

    print("Example Predicted Scores:", [f"{s:.3f}" for s in pred_scores[:10]])
    print("Example Mapped Classes:", pred_classes_names[:10])

    return pred_scores, pred_classes_names # Return names list