# evaluation.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, r2_score
import os
from typing import Dict, Tuple, List  # Added List
import numpy as np  # Added numpy for filtering

# --- Import Model Definition ---
# This assumes the model definition is accessible. If not, adjust the import path.
try:
    from Model.engagement_regression_model import EngagementRegressionModel
except ImportError:
    print("Warning: Could not import EngagementRegressionModel. Ensure it's in the correct path.")


    # Define a dummy class if needed for type hinting, or remove type hint
    class EngagementRegressionModel(nn.Module):
        pass

# --- Import Utility Functions ---
# This assumes utils are accessible. If not, adjust the import path.
try:
    from Model.utils import get_targets, map_score_to_class_idx
except ImportError:
    print("Warning: Could not import utility functions from Model.utils.")


    # Define dummy functions if needed
    def get_targets(*args, **kwargs):
        return None, None


    def map_score_to_class_idx(*args, **kwargs):
        return torch.tensor([])


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
    # Allow plotting even if mapped accuracy isn't present, just skip that part
    # if not history.get('val_accuracy_mapped'):
    #     print("Plotting skipped: Insufficient history data for accuracy curve.")
    #     return

    epochs = range(1, len(history['train_loss']) + 1)
    plt.style.use('seaborn-v0_8-darkgrid')  # Use a modern seaborn style

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
        plt.axhline(0, color='grey', lw=0.5, linestyle='--')  # Zero line for reference
        plt.legend()
        plt.grid(True)

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(loss_curve_path)
        print(f"Loss curves plot saved to {loss_curve_path}")
        plt.close()  # Close the figure to free memory

    except Exception as e:
        print(f"Error plotting loss curves: {e}")
        plt.close()  # Ensure plot is closed even if error occurs

    # --- Mapped Accuracy Plot ---
    if 'val_accuracy_mapped' in history and history['val_accuracy_mapped']:
        try:
            plt.figure(figsize=(7, 5))
            plt.plot(epochs, history['val_accuracy_mapped'], 'mo-', label='Validation Accuracy (Mapped)')
            plt.title('Validation Accuracy (Mapped from Regression)')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)  # Accuracy is typically between 0 and 1
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(acc_curve_path)
            print(f"Mapped accuracy curve plot saved to {acc_curve_path}")
            plt.close()
        except Exception as e:
            print(f"Error plotting mapped accuracy curve: {e}")
            plt.close()
    else:
        print("Skipping mapped accuracy plot: 'val_accuracy_mapped' not found in history.")


# ================================================
# === Evaluation Function ===
# ================================================
def evaluate_model(
        model: EngagementRegressionModel,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        label_to_idx_map: Dict[str, int],
        idx_to_score_map: Dict[int, float],
        idx_to_name_map: Dict[int, str],
        confusion_matrix_path: str,
        snp_index: int = 4  # Pass SNP index explicitly, defaulting to 4 based on config
) -> Tuple[float, float, float]:  # Modified return signature
    """
    Evaluates the model on the test set, reporting MSE loss, mapped multi-class metrics,
    and mapped binary classification metrics (Engaged vs. Not Engaged).
    Saves confusion matrix plots for both multi-class and binary evaluations.

    Args:
        model (EngagementRegressionModel): The trained model instance.
        test_loader (DataLoader): DataLoader for the test set.
        criterion (nn.Module): The loss function (e.g., nn.MSELoss).
        device (torch.device): The device to run evaluation on.
        label_to_idx_map (Dict[str, int]): Mapping from string labels to class indices.
        idx_to_score_map (Dict[int, float]): Mapping from class indices to regression scores.
        idx_to_name_map (Dict[int, str]): Mapping from class indices to class names.
        confusion_matrix_path (str): Base path to save confusion matrix plots.
                                     '_binary' will be appended for the binary plot.
        snp_index (int): The index corresponding to the 'SNP' class, to be excluded
                         from binary evaluation. Defaults to 4.

    Returns:
        Tuple[float, float, float]: A tuple containing:
            - final_test_loss (float): The average MSE loss on the test set.
            - final_cls_acc (float): The accuracy based on mapped multi-class predictions.
            - final_binary_acc (float): The accuracy based on mapped binary predictions
                                        (excluding SNP samples), or 0.0 if not calculable.
    """
    print("\n--- Evaluating on Test Set ---")
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    test_samples = 0
    # Lists to store predictions and targets for metric calculation
    all_pred_scores_list: List[float] = []
    all_target_scores_list: List[float] = []
    all_pred_classes_list: List[int] = []
    all_target_classes_list: List[int] = []
    final_cls_acc = 0.0  # Default value
    final_binary_acc = 0.0  # Default value

    with torch.no_grad():  # Disable gradient calculations
        test_pbar = tqdm(test_loader, desc="Testing", leave=False, ncols=100)
        for inputs, labels_dict in test_pbar:
            if not isinstance(inputs, torch.Tensor):
                print("Warning: Skipping batch, inputs are not a tensor.")
                continue
            inputs = inputs.to(device)

            # Get target scores and indices
            targets = get_targets(labels_dict, label_to_idx_map, idx_to_score_map)
            if targets is None:
                print("Warning: Skipping batch, failed to get targets.")
                continue  # Skip invalid batches
            target_scores, target_indices = targets
            target_scores = target_scores.to(device)
            # Keep target_indices on CPU for easier accumulation

            batch_size = inputs.size(0)
            test_samples += batch_size

            # Get model predictions
            pred_scores = model(inputs)  # Shape: (batch_size, 1)
            # Calculate loss for this batch
            loss = criterion(pred_scores, target_scores)
            test_loss += loss.item() * batch_size  # Accumulate total loss

            # Map predictions to classes
            # Ensure map_score_to_class_idx returns integer indices directly
            pred_classes = map_score_to_class_idx(pred_scores)  # Expects (batch_size) tensor of indices

            # Collect scores and classes (move predictions to CPU)
            all_pred_scores_list.extend(pred_scores.squeeze().cpu().tolist())
            all_target_scores_list.extend(target_scores.squeeze().cpu().tolist())
            # Ensure pred_classes and target_indices are 1D lists of integers
            all_pred_classes_list.extend(pred_classes.squeeze().cpu().tolist())
            all_target_classes_list.extend(target_indices.cpu().tolist())  # Already on CPU

    # Convert lists to numpy arrays for easier handling
    all_pred_scores = np.array(all_pred_scores_list)
    all_target_scores = np.array(all_target_scores_list)
    all_pred_classes = np.array(all_pred_classes_list)
    all_target_classes = np.array(all_target_classes_list)

    # --- Calculate and Print Overall Metrics ---
    final_test_loss = (test_loss / test_samples) if test_samples > 0 else 0
    print(f"\nTest MSE Loss: {final_test_loss:.6f}")

    # Ensure we have collected data before calculating metrics
    if len(all_target_scores) == 0 or len(all_pred_scores) == 0:
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

    # --- Multi-class Classification Metrics ---
    print("\n--- Evaluating Multi-Class Classification (Mapped) ---")
    if len(all_target_classes) == 0 or len(all_pred_classes) == 0:
        print("\nWarning: No valid classes collected. Skipping multi-class classification metrics.")
    else:
        try:
            # Determine the unique set of labels present in targets and predictions
            present_labels = sorted(list(set(all_target_classes) | set(all_pred_classes)))
            # Get corresponding names for the report and confusion matrix
            target_names = [idx_to_name_map.get(i, f"Unknown({i})") for i in present_labels]

            # Calculate accuracy
            final_cls_acc = accuracy_score(all_target_classes, all_pred_classes)
            print(f"\nTest Multi-Class Accuracy (mapped): {final_cls_acc:.4f}")

            # Generate classification report
            print("\nMulti-Class Classification Report (mapped):")
            # Use present_labels to ensure the report only includes relevant classes
            print(classification_report(all_target_classes, all_pred_classes, labels=present_labels,
                                        target_names=target_names, digits=4, zero_division=0))

            # Generate confusion matrix
            print("Multi-Class Confusion Matrix (mapped):")
            cm = confusion_matrix(all_target_classes, all_pred_classes, labels=present_labels)
            print(cm)

            # Plot Confusion Matrix
            try:
                # Adjust figure size based on the number of labels for better readability
                plt.figure(figsize=(max(7, len(present_labels) * 1.2), max(5, len(present_labels))))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=target_names, yticklabels=target_names,
                            annot_kws={"size": 8})  # Adjust font size
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title('Confusion Matrix (Mapped from Regression)')
                plt.tight_layout()
                plt.savefig(confusion_matrix_path)
                print(f"Multi-class confusion matrix plot saved to {confusion_matrix_path}")
                plt.close()
            except Exception as e_plot:
                print(f"Could not plot multi-class confusion matrix: {e_plot}")
                plt.close()  # Ensure plot is closed even if error occurs during plotting

        except Exception as e:
            print(f"Could not calculate multi-class classification metrics: {e}")
            # Ensure plot is closed if an error occurred before saving/closing
            try:
                plt.close()
            except:
                pass  # Ignore if no figure is open

    # --- Binary Classification Metrics (Engaged vs. Not Engaged, excluding SNP) ---
    print("\n--- Evaluating Binary Classification (Engaged vs. Not Engaged, Excluding SNP) ---")
    if len(all_target_classes) == 0 or len(all_pred_classes) == 0:
        print("\nWarning: No valid classes collected. Skipping binary classification metrics.")
    else:
        try:
            # Filter out SNP samples using the provided snp_index
            non_snp_mask = (all_target_classes != snp_index)

            if not np.any(non_snp_mask):
                print("Skipping binary evaluation: No non-SNP samples found in the test set.")
            else:
                # Apply mask to get data excluding SNP targets
                filtered_target_classes = all_target_classes[non_snp_mask]
                filtered_pred_classes = all_pred_classes[non_snp_mask]

                # Define mapping from original index to binary class (0: Not Engaged, 1: Engaged)
                # Indices 0 ('Not Engaged'), 1 ('Barely Engaged') -> 0
                # Indices 2 ('Engaged'), 3 ('Highly Engaged')   -> 1
                # We can use np.isin for efficient mapping
                binary_targets = np.isin(filtered_target_classes, [2, 3]).astype(int)
                binary_preds = np.isin(filtered_pred_classes, [2, 3]).astype(int)

                # Calculate binary metrics
                final_binary_acc = accuracy_score(binary_targets, binary_preds)
                print(f"\nTest Binary Accuracy (Engaged/Not Engaged, excluding SNP): {final_binary_acc:.4f}")

                # Classification Report
                binary_target_names = ['Not Engaged (Binary)', 'Engaged (Binary)']
                print("\nBinary Classification Report (excluding SNP):")
                # Use labels=[0, 1] to ensure both classes appear if expected
                present_binary_labels = sorted(list(set(binary_targets) | set(binary_preds)))
                report_target_names = [binary_target_names[i] for i in present_binary_labels]
                print(classification_report(binary_targets, binary_preds, labels=present_binary_labels,
                                            target_names=report_target_names, digits=4, zero_division=0))

                # Confusion Matrix
                print("Binary Confusion Matrix (excluding SNP):")
                # Use labels=[0, 1] to get a consistent 2x2 matrix shape
                binary_cm = confusion_matrix(binary_targets, binary_preds, labels=[0, 1])
                print(binary_cm)

                # Plot Binary Confusion Matrix
                try:
                    # Create a separate path for the binary CM plot
                    base, ext = os.path.splitext(confusion_matrix_path)
                    binary_cm_path = f"{base}_binary{ext}"

                    plt.figure(figsize=(6, 4))  # Standard size for 2x2 matrix
                    sns.heatmap(binary_cm, annot=True, fmt='d', cmap='Greens',  # Use a different colormap
                                xticklabels=binary_target_names, yticklabels=binary_target_names)
                    plt.xlabel('Predicted Label (Binary)')
                    plt.ylabel('True Label (Binary)')
                    plt.title('Binary Confusion Matrix (Engaged vs. Not Engaged)')
                    plt.tight_layout()
                    plt.savefig(binary_cm_path)
                    print(f"Binary confusion matrix plot saved to {binary_cm_path}")
                    plt.close()
                except Exception as e_plot_bin:
                    print(f"Could not plot binary confusion matrix: {e_plot_bin}")
                    plt.close()  # Ensure plot is closed

        except Exception as e_bin:
            print(f"\n!!! ERROR during binary classification evaluation: {e_bin} !!!")
            # Ensure plot is closed if error occurs
            try:
                plt.close()
            except:
                pass  # Ignore if no figure is open

    # Return main scalar metrics including binary accuracy
    return final_test_loss, final_cls_acc, final_binary_acc
