import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple
from Model.engagement_regression_model import EngagementRegressionModel
from Model.utils import map_score_to_class_idx


# ================================================
# === Prediction Function ===
# ================================================
def predict_engagement(
    model: EngagementRegressionModel,
    data_loader: DataLoader,
    device: torch.device,
    idx_to_name_map: Dict[int, str]
    ) -> Tuple[List[float], List[str]]:
    """
    Predicts engagement scores and mapped class names for a given data loader.

    Args:
        model (EngagementRegressionModel): The trained model instance.
        data_loader (DataLoader): DataLoader containing the input data (expects to yield inputs first).
        device (torch.device): The device to run prediction on.
        idx_to_name_map (Dict[int, str]): Mapping from class indices to class names.

    Returns:
        Tuple[List[float], List[str]]: A tuple containing:
            - pred_scores (List[float]): A list of predicted regression scores.
            - pred_classes (List[str]): A list of predicted class names corresponding to the scores.
    """
    model.eval() # Set model to evaluation mode
    pred_scores = []
    pred_classes = []
    print("\n--- Making Predictions ---")

    with torch.no_grad(): # Disable gradient calculations
        pbar = tqdm(data_loader, desc="Predicting", leave=False, ncols=100)
        # Assume loader yields (data, label) or just data; we only need data
        for batch in pbar:
             # Handle cases where loader yields tuples or just tensors
             if isinstance(batch, (list, tuple)):
                 inputs = batch[0]
             else:
                 inputs = batch

             # Ensure input is a tensor
             if not isinstance(inputs, torch.Tensor):
                 # print(f"Warning: Skipping non-tensor input of type {type(inputs)}") # Optional warning
                 continue

             inputs = inputs.to(device)
             outputs = model(inputs) # Get model predictions, Shape: (batch_size, 1)

             # Squeeze to remove the last dimension for easier processing, Shape: (batch_size)
             scores = outputs.squeeze(-1)
             # Map scores to class indices
             classes_idx = map_score_to_class_idx(scores) # Shape: (batch_size)

             # Store results (move tensors to CPU and convert to Python lists/types)
             pred_scores.extend(scores.cpu().numpy().tolist())
             # Map class indices back to names using the provided map
             pred_classes.extend([idx_to_name_map.get(c.item(), "Unknown") for c in classes_idx])

    # Print examples of the predictions
    print("Example Predicted Scores:", [f"{s:.3f}" for s in pred_scores[:10]])
    print("Example Mapped Classes:", pred_classes[:10])

    return pred_scores, pred_classes
