import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, r2_score
import os
from typing import Dict, Tuple
from Model.engagement_regression_model import EngagementRegressionModel
from Model.utils import get_targets, map_score_to_class_idx


# ================================================
# === Plotting Function ===
# ================================================
def plot_training_history(history: Dict, loss_curve_path: str, acc_curve_path: str):
    """
    Plots training/validation loss and mapped accuracy curves. Saves the plots to specified paths.

    Args:
        history (Dict): Dictionary containing training history
                        (expects 'train_loss', 'val_loss', 'val_accuracy_mapped').
        loss_curve_path (str): Path to save the loss curves plot.
        acc_curve_path (str): Path to save the accuracy curve plot.
    """
    # Check if history data is sufficient
    if not history or not history.get('train_loss') or not history.get('val_loss'):
        print("Plotting skipped: Insufficient history data for loss curves.")
        return
    if not history.get('val_accuracy_mapped'):
        print("Plotting skipped: Insufficient history data for accuracy curve.")
        return


    epochs = range(1, len(history['train_loss']) + 1)
    plt.style.use('seaborn-v0_8-darkgrid') # Use a modern seaborn style

    # --- Loss Plot ---
    try:
        plt.figure(figsize=(12, 5))

        # Subplot 1: Training vs Validation Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss (MSE)')
        plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)

        # Subplot 2: Difference between Validation and Training Loss
        plt.subplot(1, 2, 2)
        loss_diff = [v - t for v, t in zip(history['val_loss'], history['train_loss'])]
        plt.plot(epochs, loss_diff, 'go-', label='Validation - Training Loss')
        plt.title('Loss Difference (Val - Train)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Difference')
        plt.axhline(0, color='grey', lw=0.5, linestyle='--') # Zero line for reference
        plt.legend()
        plt.grid(True)

        plt.tight_layout() # Adjust layout to prevent overlap
        plt.savefig(loss_curve_path)
        print(f"Loss curves plot saved to {loss_curve_path}")
        plt.close() # Close the figure to free memory

    except Exception as e:
        print(f"Error plotting loss curves: {e}")
        plt.close() # Ensure plot is closed even if error occurs

    # --- Mapped Accuracy Plot ---
    if 'val_accuracy_mapped' in history and history['val_accuracy_mapped']:
        try:
            plt.figure(figsize=(7, 5))
            plt.plot(epochs, history['val_accuracy_mapped'], 'mo-', label='Validation Accuracy (Mapped)')
            plt.title('Validation Accuracy (Mapped from Regression)')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1) # Accuracy is typically between 0 and 1
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(acc_curve_path)
            print(f"Mapped accuracy curve plot saved to {acc_curve_path}")
            plt.close()
        except Exception as e:
            print(f"Error plotting mapped accuracy curve: {e}")
            plt.close()

# ================================================
# === Evaluation Function ===
# ================================================
def evaluate_model(
    model: EngagementRegressionModel,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_to_idx_map: Dict[str, int], # Pass maps explicitly
    idx_to_score_map: Dict[int, float], # Pass maps explicitly
    idx_to_name_map: Dict[int, str],   # Pass maps explicitly
    confusion_matrix_path: str
    ) -> Tuple[float, float]:
    """
    Evaluates the model on the test set, reporting MSE loss and mapped classification metrics.
    Saves a confusion matrix plot.

    Args:
        model (EngagementRegressionModel): The trained model instance.
        test_loader (DataLoader): DataLoader for the test set.
        criterion (nn.Module): The loss function (e.g., nn.MSELoss).
        device (torch.device): The device to run evaluation on.
        label_to_idx_map (Dict[str, int]): Mapping from string labels to class indices.
        idx_to_score_map (Dict[int, float]): Mapping from class indices to regression scores.
        idx_to_name_map (Dict[int, str]): Mapping from class indices to class names.
        confusion_matrix_path (str): Path to save the confusion matrix plot.

    Returns:
        Tuple[float, float]: A tuple containing:
            - final_test_loss (float): The average MSE loss on the test set.
            - final_cls_acc (float): The accuracy based on mapped class predictions.
    """
    print("\n--- Evaluating on Test Set ---")
    model.eval() # Set model to evaluation mode
    test_loss = 0.0
    test_samples = 0
    # Lists to store predictions and targets for metric calculation
    all_pred_scores = []
    all_target_scores = []
    all_pred_classes = []
    all_target_classes = []
    final_cls_acc = 0.0 # Default value in case metrics can't be calculated

    with torch.no_grad(): # Disable gradient calculations
        test_pbar = tqdm(test_loader, desc="Testing", leave=False, ncols=100)
        for inputs, labels_dict in test_pbar:
            if not isinstance(inputs, torch.Tensor): continue
            inputs = inputs.to(device)

            # Get target scores and indices
            targets = get_targets(labels_dict, label_to_idx_map, idx_to_score_map)
            if targets is None: continue # Skip invalid batches
            target_scores, target_indices = targets
            target_scores = target_scores.to(device)
            # Keep target_indices on CPU for easier accumulation with sklearn metrics later

            batch_size = inputs.size(0)
            test_samples += batch_size

            # Get model predictions
            pred_scores = model(inputs) # Shape: (batch_size, 1)
            # Calculate loss for this batch
            loss = criterion(pred_scores, target_scores)
            test_loss += loss.item() * batch_size # Accumulate total loss

            # Map predictions to classes
            pred_classes = map_score_to_class_idx(pred_scores) # Shape: (batch_size) or (batch_size, 1)

            # Collect scores and classes (move predictions to CPU)
            all_pred_scores.extend(pred_scores.squeeze().cpu().numpy())
            all_target_scores.extend(target_scores.squeeze().cpu().numpy())
            all_pred_classes.extend(pred_classes.squeeze().cpu().numpy())
            all_target_classes.extend(target_indices.cpu().numpy()) # Already on CPU

    # --- Calculate and Print Overall Metrics ---
    final_test_loss = test_loss / test_samples if test_samples > 0 else 0
    print(f"\nTest MSE Loss: {final_test_loss:.6f}")

    # Ensure we have collected data before calculating metrics
    if not all_target_scores or not all_pred_scores:
        print("\nWarning: No valid scores collected. Skipping regression metrics.")
    else:
        # --- Regression Metrics ---
        try:
            mae = mean_absolute_error(all_target_scores, all_pred_scores)
            r2 = r2_score(all_target_scores, all_pred_scores)
            print(f"Test MAE (Regression): {mae:.4f}")
            print(f"Test R^2 Score (Regression): {r2:.4f}")
        except Exception as e:
            print(f"Could not calculate regression metrics: {e}")

    if not all_target_classes or not all_pred_classes:
         print("\nWarning: No valid classes collected. Skipping classification metrics.")
    else:
        # --- Mapped Classification Metrics ---
        try:
            # Determine the unique set of labels present in targets and predictions
            present_labels = sorted(list(set(all_target_classes) | set(all_pred_classes)))
            # Get corresponding names for the report and confusion matrix
            target_names = [idx_to_name_map.get(i, f"Unknown({i})") for i in present_labels]

            # Calculate accuracy
            final_cls_acc = accuracy_score(all_target_classes, all_pred_classes)
            print(f"\nTest Classification Accuracy (mapped): {final_cls_acc:.4f}")

            # Generate classification report
            print("\nClassification Report (mapped):")
            # Use present_labels to ensure the report only includes relevant classes
            print(classification_report(all_target_classes, all_pred_classes, labels=present_labels, target_names=target_names, digits=4, zero_division=0))

            # Generate confusion matrix
            print("Confusion Matrix (mapped):")
            cm = confusion_matrix(all_target_classes, all_pred_classes, labels=present_labels)
            print(cm)

            # Plot Confusion Matrix
            try:
                # Adjust figure size based on the number of labels for better readability
                plt.figure(figsize=(max(7, len(present_labels)), max(5, len(present_labels)-1)))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=target_names, yticklabels=target_names)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title('Confusion Matrix (Mapped from Regression)')
                plt.tight_layout()
                plt.savefig(confusion_matrix_path)
                print(f"Confusion matrix plot saved to {confusion_matrix_path}")
                plt.close()
            except Exception as e_plot:
                print(f"Could not plot confusion matrix: {e_plot}")
                plt.close() # Ensure plot is closed even if error occurs during plotting

        except Exception as e:
            print(f"Could not calculate classification metrics: {e}")
            # Ensure plot is closed if an error occurred before saving/closing
            plt.close()

    return final_test_loss, final_cls_acc # Return main scalar metrics
