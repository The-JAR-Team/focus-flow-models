import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import copy
from typing import Tuple, Dict
from Model.engagement_regression_model import EngagementRegressionModel
from Model.utils import get_targets, map_score_to_class_idx


# ================================================
# === Training Function ===
# ================================================
def train_model(
    model: EngagementRegressionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_path_pth: str,
    save_best_pth: bool,
    label_to_idx_map: Dict[str, int], # Pass maps explicitly
    idx_to_score_map: Dict[int, float]  # Pass maps explicitly
    ) -> Tuple[EngagementRegressionModel, Dict]:
    """
    Trains the regression model, optionally saving the best .pth state based on validation loss.

    Args:
        model (EngagementRegressionModel): The model instance to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): The loss function (e.g., nn.MSELoss).
        optimizer (optim.Optimizer): The optimizer (e.g., AdamW).
        num_epochs (int): The total number of epochs to train for.
        device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').
        save_path_pth (str): Path to save the best PyTorch model state dictionary.
        save_best_pth (bool): Flag indicating whether to save the best model .pth file.
        label_to_idx_map (Dict[str, int]): Mapping from string labels to class indices.
        idx_to_score_map (Dict[int, float]): Mapping from class indices to regression scores.


    Returns:
        Tuple[EngagementRegressionModel, Dict]: A tuple containing:
            - final_model (EngagementRegressionModel): The model with the best loaded weights (or last epoch's weights).
            - history (Dict): A dictionary containing training and validation loss/accuracy history.
    """
    best_val_loss = float('inf')
    best_model_state_dict = None # Store best state in memory if saving to file is disabled or fails
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy_mapped': []}

    print("\n--- Starting Training (Regression [0, 1]) ---")
    if save_best_pth:
        print(f"Best PyTorch model (.pth) will be saved to: {save_path_pth}")
    else:
        print("PyTorch model saving (.pth) is disabled during training.")

    start_train_time = time.time()
    # --- Epoch Loop ---
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_start_time = time.time()

        # --- Training Phase ---
        model.train() # Set model to training mode
        running_loss = 0.0
        total_samples_train = 0
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc="  Training", leave=False, ncols=100)
        for inputs, labels_dict in train_pbar:
            # Basic check for tensor type, assuming get_dataloader yields tuples (data, labels_dict)
            if not isinstance(inputs, torch.Tensor):
                # print(f"Warning: Skipping non-tensor input of type {type(inputs)}") # Optional warning
                continue
            inputs = inputs.to(device)

            # Get target scores and indices using the utility function
            targets = get_targets(labels_dict, label_to_idx_map, idx_to_score_map)
            if targets is None:
                # print("Warning: Skipping batch due to invalid labels.") # Optional warning
                continue # Skip batch if labels are invalid

            target_scores, _ = targets # We only need scores for MSE loss
            target_scores = target_scores.to(device)

            batch_size = inputs.size(0)
            total_samples_train += batch_size

            # Standard training steps
            optimizer.zero_grad()
            outputs = model(inputs) # Forward pass, Shape: (batch_size, 1)
            loss = criterion(outputs, target_scores) # Calculate loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights

            running_loss += loss.item() * batch_size # Accumulate loss correctly (scale by batch size)
            # Update progress bar postfix
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Calculate average training loss for the epoch
        epoch_train_loss = running_loss / total_samples_train if total_samples_train > 0 else 0
        history['train_loss'].append(epoch_train_loss)

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        val_corrects_mapped = 0
        total_val_samples = 0
        total_val_valid_samples_acc = 0 # Samples used for accuracy calculation

        with torch.no_grad(): # Disable gradient calculations for validation
            # Progress bar for validation
            val_pbar = tqdm(val_loader, desc="  Validation", leave=False, ncols=100)
            for inputs, labels_dict in val_pbar:
                if not isinstance(inputs, torch.Tensor): continue
                inputs = inputs.to(device)

                # Get targets for validation loss and mapped accuracy
                targets = get_targets(labels_dict, label_to_idx_map, idx_to_score_map)
                if targets is None: continue

                target_scores, target_indices = targets # Need indices for mapped accuracy
                target_scores = target_scores.to(device)
                target_indices = target_indices.to(device) # Move indices to device for comparison

                batch_size = inputs.size(0)
                total_val_samples += batch_size

                # Predict scores
                pred_scores = model(inputs) # Shape: (batch_size, 1)
                # Calculate validation loss
                loss = criterion(pred_scores, target_scores)
                val_loss += loss.item() * batch_size

                # Calculate Mapped Accuracy
                pred_classes = map_score_to_class_idx(pred_scores) # Map scores to classes
                # Compare mapped predictions to original target class indices (on the same device)
                # .squeeze() handles cases where pred_classes might be [batch, 1]
                val_corrects_mapped += torch.sum(pred_classes.squeeze() == target_indices.squeeze())
                total_val_valid_samples_acc += batch_size # Count all valid samples for accuracy average

        # Calculate average validation loss and accuracy for the epoch
        epoch_val_loss = val_loss / total_val_samples if total_val_samples > 0 else float('inf')
        # Ensure division by zero doesn't occur if no valid samples
        epoch_val_acc_mapped = val_corrects_mapped.double() / total_val_valid_samples_acc if total_val_valid_samples_acc > 0 else 0.0
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy_mapped'].append(epoch_val_acc_mapped.item()) # Store accuracy as float

        epoch_end_time = time.time()
        # --- Print Epoch Summary ---
        print(f"  Epoch {epoch+1} Summary:")
        print(f"    Time:         {(epoch_end_time - epoch_start_time):.2f}s")
        print(f"    Train Loss:   {epoch_train_loss:.6f}") # More precision for loss
        print(f"    Val Loss:     {epoch_val_loss:.6f}")
        print(f"    Val Acc (map):{epoch_val_acc_mapped:.4f}")

        # --- Save Best Model Check ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            print(f"    * New best Val Loss: {best_val_loss:.6f}. ", end="")
            if save_best_pth:
                try:
                    # Save only the model's state dictionary
                    torch.save(model.state_dict(), save_path_pth)
                    print(f"Model saved to {save_path_pth}")
                    best_model_state_dict = None # Clear memory state if saved to file successfully
                except Exception as e:
                    print(f"\n      ERROR saving model state dict: {e}")
                    print("      Keeping best state in memory instead.")
                    # Fallback to keeping the state dict in memory
                    best_model_state_dict = copy.deepcopy(model.state_dict())
            else:
                # If saving is disabled, always keep the best state in memory
                print("(Saving disabled, keeping state in memory)")
                best_model_state_dict = copy.deepcopy(model.state_dict())

    # --- End of Epoch Loop ---
    end_train_time = time.time()
    print('\n--- Training Finished ---')
    print(f"Total Training Time: {(end_train_time - start_train_time):.2f}s")
    print(f"Best Validation Loss achieved: {best_val_loss:.6f}")

    # --- Load Best State into the final model ---
    final_model = model # Start with the model from the last epoch
    # Prioritize loading from file if saving was enabled and file exists
    if save_best_pth and os.path.exists(save_path_pth):
        print(f"Loading best model weights from {save_path_pth}")
        try:
            final_model.load_state_dict(torch.load(save_path_pth, map_location=device))
        except Exception as e:
            print(f"Error loading saved state from {save_path_pth}: {e}.")
            if best_model_state_dict:
                 print("Attempting to load best state from memory...")
                 try:
                     final_model.load_state_dict(best_model_state_dict)
                     print("Loaded best state from memory successfully.")
                 except Exception as e_mem:
                     print(f"Error loading state from memory: {e_mem}. Returning last epoch model.")
            else:
                print("Warning: Best model file not found/failed to load and no state in memory. Returning model from "
                      "last epoch.")
    # If saving to file was disabled OR failed, but we have a state in memory
    elif best_model_state_dict:
        print("Loading best model state from memory.")
        try:
            final_model.load_state_dict(best_model_state_dict)
        except Exception as e:
            print(f"Error loading state from memory: {e}. Returning last epoch model.")
    # If saving was disabled and no state was ever stored in memory (e.g., error during first epoch save)
    else:
        print("No best model state available (saving disabled or failed/interrupted early). Returning model from the "
              "last epoch.")

    return final_model, history
