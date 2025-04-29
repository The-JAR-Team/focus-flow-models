import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import traceback
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any, List
import time
import copy
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import onnx
import onnxruntime
from Preprocess.Pipeline.InspectData import get_dataloader


# ================================================
# === Configuration ===
# ================================================
CONFIG_PATH = "./Preprocess/Pipeline/Pipelines/configs/ENGAGENET_10fps_quality95_randdist.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model & Training Hyperparameters ---
INPUT_DIM = 478 * 3 # Example: (num_landmarks * coordinates) - Adjust if needed
HIDDEN_DIM = 256
NUM_GRU_LAYERS = 2
DROPOUT_RATE = 0.4
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 50

# --- Saving & Loading ---
# Base name for saved files (paths will be constructed)
MODEL_BASE_NAME = "engagement_model_regression_0_1"
SAVE_DIR = "." # Directory to save models and plots

# Construct full paths
MODEL_SAVE_PATH_PTH = os.path.join(SAVE_DIR, f"{MODEL_BASE_NAME}.pth")
MODEL_SAVE_PATH_ONNX = os.path.join(SAVE_DIR, f"{MODEL_BASE_NAME}.onnx")

SAVE_BEST_MODEL_PTH = True # Save best model state dict during training?
SAVE_FINAL_MODEL_ONNX = True # Save the final best model as ONNX after training?
LOAD_SAVED_STATE = False # Attempt to load MODEL_SAVE_PATH_PTH before training?

# --- Print Configuration ---
print("--- Configuration ---")
print(f"Device: {DEVICE}")
print(f"Config Path: {CONFIG_PATH}")
print(f"Input Dim: {INPUT_DIM}")
print(f"Load Saved State: {LOAD_SAVED_STATE}")
print(f"Save Best PyTorch Model (.pth): {SAVE_BEST_MODEL_PTH}")
if SAVE_BEST_MODEL_PTH or LOAD_SAVED_STATE:
    print(f"  PyTorch Model Path: {MODEL_SAVE_PATH_PTH}")
print(f"Save Final ONNX Model (.onnx): {SAVE_FINAL_MODEL_ONNX}")
if SAVE_FINAL_MODEL_ONNX:
    print(f"  ONNX Model Path: {MODEL_SAVE_PATH_ONNX}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Target Score Range: [0.0, 1.0]")
print("-------------------")

# ================================================
# === Mappings ===
# ================================================
LABEL_TO_IDX_MAP = {
    'Not Engaged': 0, 'Barely Engaged': 1, 'Engaged': 2, 'Highly Engaged': 3,
    'not engaged': 0, 'not-engaged': 0, 'Not-Engaged': 0,
    'barely engaged': 1, 'barely-engaged': 1, 'Barely-engaged': 1,
    'highly engaged': 3, 'highly-engaged': 3, 'Highly-Engaged': 3,
    'snp(subject not present)': 4, 'SNP(Subject Not Present)': 4, 'SNP': 4,
}
IDX_TO_SCORE_MAP = {4: 0.0, 0: 0.25, 1: 0.5, 2: 0.75, 3: 1.0} # SNP mapped to 0.0 score
IDX_TO_NAME_MAP = {0: 'Not Engaged', 1: 'Barely Engaged', 2: 'Engaged', 3: 'Highly Engaged', 4: 'SNP'}
print("Label -> Index -> Score Mapping Initialized.")
# ================================================


# ================================================
# === Model Definition ===
# ================================================
class EngagementRegressionModel(nn.Module):
    """ GRU-based model for engagement regression. """
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.5, bidirectional=True):
        super().__init__()
        self.input_dim = input_dim
        self.bidirectional = bidirectional

        self.frame_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)

        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(gru_output_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.output_activation = nn.Sigmoid() # Output between 0 and 1

    def forward(self, x):
        # Input shape: (batch, seq_len, num_landmarks, coords)
        batch_size, seq_len, num_landmarks, coords = x.shape
        # Reshape: (batch, seq_len, num_landmarks * coords)
        x = x.reshape(batch_size, seq_len, -1)

        if x.shape[2] != self.input_dim:
             raise ValueError(f"Input dim mismatch: Expected {self.input_dim}, Got {x.shape[2]}")

        x = self.frame_norm(x)
        gru_out, hn = self.gru(x) # gru_out: (batch, seq_len, hidden*dirs), hn: (num_layers*dirs, batch, hidden)

        # Use the hidden state of the last time step
        if self.bidirectional:
            # Concatenate the last hidden states from forward and backward directions
            last_hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        else:
            last_hidden = hn[-1,:,:]

        out = self.dropout(last_hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.output_activation(out) # Ensure output is in [0, 1]
        return out
# ================================================


# ================================================
# === Utility Functions ===
# ================================================
def get_targets(label_batch: Dict[str, List[Any]], label_to_idx_map: Dict[str, int], idx_to_score_map: Dict[int, float]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """ Converts string labels from batch dict to score and index tensors. """
    if 'engagement_string' not in label_batch:
        # print("Warning: 'engagement_string' key missing in label batch.")
        return None # Skip batch if key is missing

    string_labels = label_batch['engagement_string']
    target_scores = []
    target_indices = []

    for lbl in string_labels:
        processed_lbl = str(lbl).strip() # Handle potential non-string types and whitespace
        class_idx = label_to_idx_map.get(processed_lbl)

        if class_idx is None:
            # print(f"Warning: Unknown label encountered: '{lbl}'. Skipping batch.")
            return None # Skip entire batch if one label is unknown

        score = idx_to_score_map.get(class_idx)
        if score is None:
            # Should not happen if maps are consistent, but good practice to check
            # print(f"Warning: No score defined for index {class_idx} ('{processed_lbl}'). Skipping batch.")
            return None # Skip batch if score mapping fails

        target_indices.append(class_idx)
        target_scores.append(score)

    # Convert lists to tensors
    # Ensure scores are float and have shape (batch_size, 1) for MSELoss
    scores_tensor = torch.tensor(target_scores, dtype=torch.float).unsqueeze(1)
    # Indices are long integers for potential classification use
    indices_tensor = torch.tensor(target_indices, dtype=torch.long)

    return scores_tensor, indices_tensor

def map_score_to_class_idx(score_tensor: torch.Tensor) -> torch.Tensor:
    """ Maps continuous regression scores [0, 1] back to discrete class indices (0-4). """
    # Thresholds centered between target scores (0.0, 0.25, 0.5, 0.75, 1.0)
    # SNP(4): [0.0, 0.125) -> Mapped from score 0.0
    # Not(0): [0.125, 0.375) -> Mapped from score 0.25
    # Barely(1):[0.375, 0.625) -> Mapped from score 0.5
    # Engaged(2):[0.625, 0.875) -> Mapped from score 0.75
    # Highly(3):[0.875, 1.0] -> Mapped from score 1.0
    thresholds = [0.125, 0.375, 0.625, 0.875]
    # Ensure tensor is flat for thresholding logic, handle potential extra dim
    score_tensor_flat = score_tensor.squeeze()

    # Initialize predictions with a value indicating SNP (class 4)
    # Use torch.where for efficient conditional assignment
    preds = torch.full_like(score_tensor_flat, 4, dtype=torch.long) # Default to SNP (index 4)

    # Assign classes based on thresholds
    preds = torch.where((score_tensor_flat >= thresholds[0]) & (score_tensor_flat < thresholds[1]), 0, preds) # Not Engaged
    preds = torch.where((score_tensor_flat >= thresholds[1]) & (score_tensor_flat < thresholds[2]), 1, preds) # Barely Engaged
    preds = torch.where((score_tensor_flat >= thresholds[2]) & (score_tensor_flat < thresholds[3]), 2, preds) # Engaged
    preds = torch.where(score_tensor_flat >= thresholds[3], 3, preds) # Highly Engaged

    # Restore original shape if input was (batch, 1)
    if score_tensor.dim() > 1 and score_tensor.shape[-1] == 1 :
        preds = preds.unsqueeze(-1)

    return preds
# ================================================


# ================================================
# === Training Function ===
# ================================================
def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device,
    save_path_pth: str, save_best_pth: bool
    ) -> Tuple[nn.Module, Dict]:
    """ Trains the regression model, optionally saving the best .pth state based on validation loss. """
    best_val_loss = float('inf')
    best_model_state_dict = None # Store best state in memory if saving to file is disabled or fails
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy_mapped': []}

    print("\n--- Starting Training (Regression [0, 1]) ---")
    if save_best_pth:
        print(f"Best PyTorch model (.pth) will be saved to: {save_path_pth}")
    else:
        print("PyTorch model saving (.pth) is disabled during training.")

    start_train_time = time.time()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_start_time = time.time()

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        total_samples_train = 0
        train_pbar = tqdm(train_loader, desc="  Training", leave=False, ncols=100)
        for inputs, labels_dict in train_pbar:
            # Basic check for tensor type, assuming get_dataloader yields tuples
            if not isinstance(inputs, torch.Tensor): continue
            inputs = inputs.to(device)
            targets = get_targets(labels_dict, LABEL_TO_IDX_MAP, IDX_TO_SCORE_MAP)
            if targets is None: continue # Skip batch if labels are invalid
            target_scores, _ = targets
            target_scores = target_scores.to(device)

            batch_size = inputs.size(0)
            total_samples_train += batch_size

            optimizer.zero_grad()
            outputs = model(inputs) # Shape: (batch_size, 1)
            loss = criterion(outputs, target_scores)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size # Accumulate loss correctly
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train_loss = running_loss / total_samples_train if total_samples_train > 0 else 0
        history['train_loss'].append(epoch_train_loss)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects_mapped = 0
        total_val_samples = 0
        total_val_valid_samples_acc = 0 # Samples used for accuracy calculation

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="  Validation", leave=False, ncols=100)
            for inputs, labels_dict in val_pbar:
                if not isinstance(inputs, torch.Tensor): continue
                inputs = inputs.to(device)
                targets = get_targets(labels_dict, LABEL_TO_IDX_MAP, IDX_TO_SCORE_MAP)
                if targets is None: continue
                target_scores, target_indices = targets # Need indices for mapped accuracy
                target_scores = target_scores.to(device)
                target_indices = target_indices.to(device) # For comparison

                batch_size = inputs.size(0)
                total_val_samples += batch_size

                pred_scores = model(inputs) # Shape: (batch_size, 1)
                loss = criterion(pred_scores, target_scores)
                val_loss += loss.item() * batch_size

                # Calculate Mapped Accuracy
                pred_classes = map_score_to_class_idx(pred_scores) # Shape: (batch_size) or (batch_size, 1)
                # Compare mapped predictions to original target class indices
                val_corrects_mapped += torch.sum(pred_classes.squeeze() == target_indices)
                total_val_valid_samples_acc += batch_size # Count all valid samples

        epoch_val_loss = val_loss / total_val_samples if total_val_samples > 0 else float('inf')
        epoch_val_acc_mapped = val_corrects_mapped.double() / total_val_valid_samples_acc if total_val_valid_samples_acc > 0 else 0.0
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy_mapped'].append(epoch_val_acc_mapped.item()) # Store as float

        epoch_end_time = time.time()
        # --- Print Epoch Summary ---
        print(f"  Epoch {epoch+1} Summary:")
        print(f"    Time:         {(epoch_end_time - epoch_start_time):.2f}s")
        print(f"    Train Loss:   {epoch_train_loss:.6f}")
        print(f"    Val Loss:     {epoch_val_loss:.6f}")
        print(f"    Val Acc (map):{epoch_val_acc_mapped:.4f}")

        # --- Save Best Model Check ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            print(f"    * New best Val Loss: {best_val_loss:.6f}. ", end="")
            if save_best_pth:
                try:
                    torch.save(model.state_dict(), save_path_pth)
                    print(f"Model saved to {save_path_pth}")
                    best_model_state_dict = None # Clear memory state if saved to file
                except Exception as e:
                    print(f"\n      ERROR saving model state dict: {e}")
                    print("      Keeping best state in memory instead.")
                    best_model_state_dict = copy.deepcopy(model.state_dict()) # Fallback to memory
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
    if save_best_pth and os.path.exists(save_path_pth):
        print(f"Loading best model weights from {save_path_pth}")
        try:
            final_model.load_state_dict(torch.load(save_path_pth, map_location=device))
        except Exception as e:
            print(f"Error loading saved state from {save_path_pth}: {e}.")
            if best_model_state_dict: # If file loading failed but we have memory state
                 print("Attempting to load best state from memory...")
                 try:
                     final_model.load_state_dict(best_model_state_dict)
                     print("Loaded best state from memory successfully.")
                 except Exception as e_mem:
                     print(f"Error loading state from memory: {e_mem}. Returning last epoch model.")
            else: # No file and no memory state (should only happen if saving failed AND was disabled)
                print("Warning: Best model file not found and no state in memory. Returning model from last epoch.")
    elif best_model_state_dict: # If saving to file was disabled or failed, but we have memory state
        print("Loading best model state from memory.")
        try:
            final_model.load_state_dict(best_model_state_dict)
        except Exception as e:
            print(f"Error loading state from memory: {e}. Returning last epoch model.")
    else: # No file saved, saving disabled, no memory state (e.g., error during first epoch save)
        print("No best model state available (saving disabled or failed). Returning model from the last epoch.")

    return final_model, history
# ================================================


# ================================================
# === Plotting Function ===
# ================================================
def plot_training_history(history: Dict, save_dir: str = "."):
    """ Plots training/validation loss and mapped accuracy curves. """
    if not history or not history.get('train_loss') or not history.get('val_loss'):
        print("Plotting skipped: Insufficient history data.")
        return

    epochs = range(1, len(history['train_loss']) + 1)
    plt.style.use('seaborn-v0_8-darkgrid') # Use a modern seaborn style

    try:
        # --- Loss Plot ---
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss (MSE)')
        plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)

        # --- Loss Difference Plot ---
        plt.subplot(1, 2, 2)
        loss_diff = [v - t for v, t in zip(history['val_loss'], history['train_loss'])]
        plt.plot(epochs, loss_diff, 'go-', label='Validation - Training Loss')
        plt.title('Loss Difference (Val - Train)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Difference')
        plt.axhline(0, color='grey', lw=0.5, linestyle='--') # Zero line for reference
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "loss_curves.png")
        plt.savefig(plot_path)
        print(f"Loss curves plot saved to {plot_path}")
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
            plt.ylim(0, 1) # Accuracy is between 0 and 1
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(save_dir, "mapped_accuracy_curve.png")
            plt.savefig(plot_path)
            print(f"Mapped accuracy curve plot saved to {plot_path}")
            plt.close()
        except Exception as e:
            print(f"Error plotting mapped accuracy curve: {e}")
            plt.close()
# ================================================


# ================================================
# === Evaluation Function ===
# ================================================
def evaluate_model(model, test_loader, criterion, device, idx_to_name_map):
    """ Evaluates the model on the test set, reporting MSE loss and mapped classification metrics. """
    print("\n--- Evaluating on Test Set ---")
    model.eval()
    test_loss = 0.0
    test_samples = 0
    all_pred_scores = []
    all_target_scores = []
    all_pred_classes = []
    all_target_classes = []
    final_cls_acc = 0.0 # Default value

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing", leave=False, ncols=100)
        for inputs, labels_dict in test_pbar:
            if not isinstance(inputs, torch.Tensor): continue
            inputs = inputs.to(device)
            targets = get_targets(labels_dict, LABEL_TO_IDX_MAP, IDX_TO_SCORE_MAP)
            if targets is None: continue
            target_scores, target_indices = targets
            target_scores = target_scores.to(device)
            # target_indices remain on CPU for sklearn metrics

            batch_size = inputs.size(0)
            test_samples += batch_size

            pred_scores = model(inputs) # Shape: (batch_size, 1)
            loss = criterion(pred_scores, target_scores)
            test_loss += loss.item() * batch_size

            # Map predictions to classes and collect all data for metrics
            pred_classes = map_score_to_class_idx(pred_scores) # Shape: (batch_size) or (batch_size, 1)

            all_pred_scores.extend(pred_scores.squeeze().cpu().numpy())
            all_target_scores.extend(target_scores.squeeze().cpu().numpy())
            all_pred_classes.extend(pred_classes.squeeze().cpu().numpy())
            all_target_classes.extend(target_indices.cpu().numpy()) # Already on CPU

    # --- Calculate and Print Metrics ---
    final_test_loss = test_loss / test_samples if test_samples > 0 else 0
    print(f"\nTest MSE Loss: {final_test_loss:.6f}")

    # --- Regression Metrics ---
    try:
        mae = mean_absolute_error(all_target_scores, all_pred_scores)
        r2 = r2_score(all_target_scores, all_pred_scores)
        print(f"Test MAE (Regression): {mae:.4f}")
        print(f"Test R^2 Score (Regression): {r2:.4f}")
    except Exception as e:
        print(f"Could not calculate regression metrics: {e}")


    # --- Mapped Classification Metrics ---
    try:
        # Ensure labels used in metrics match the actual data present
        present_labels = sorted(list(set(all_target_classes) | set(all_pred_classes)))
        target_names = [idx_to_name_map.get(i, f"Unknown({i})") for i in present_labels]

        final_cls_acc = accuracy_score(all_target_classes, all_pred_classes)
        print(f"\nTest Classification Accuracy (mapped): {final_cls_acc:.4f}")

        print("\nClassification Report (mapped):")
        # Use present_labels for report and confusion matrix
        print(classification_report(all_target_classes, all_pred_classes, labels=present_labels, target_names=target_names, digits=4, zero_division=0))

        print("Confusion Matrix (mapped):")
        cm = confusion_matrix(all_target_classes, all_pred_classes, labels=present_labels)
        print(cm)

        # Plot Confusion Matrix if possible
        try:
            plt.figure(figsize=(max(7, len(present_labels)), max(5, len(present_labels)-1))) # Adjust size
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=target_names, yticklabels=target_names)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix (Mapped from Regression)')
            plt.tight_layout()
            cm_path = os.path.join(SAVE_DIR, "confusion_matrix_regression_mapped.png")
            plt.savefig(cm_path)
            print(f"Confusion matrix plot saved to {cm_path}")
            plt.close()
        except Exception as e_plot:
            print(f"Could not plot confusion matrix: {e_plot}")
            plt.close() # Ensure plot is closed

    except Exception as e:
        print(f"Could not calculate classification metrics: {e}")
        plt.close() # Ensure plot is closed if error occurred before saving


    return final_test_loss, final_cls_acc # Return main scalar metrics
# ================================================


# ================================================
# === Prediction Function ===
# ================================================
def predict_engagement(model, data_loader, device, idx_to_name_map):
    """ Predicts engagement scores and mapped classes for a given data loader. """
    model.eval()
    pred_scores = []
    pred_classes = []
    print("\n--- Making Predictions ---")
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Predicting", leave=False, ncols=100)
        for inputs, _ in pbar: # Assume loader yields (data, label) or just data
             if not isinstance(inputs, torch.Tensor): continue
             inputs = inputs.to(device)
             outputs = model(inputs) # Shape: (batch_size, 1)

             scores = outputs.squeeze(-1) # Shape: (batch_size)
             classes = map_score_to_class_idx(scores) # Shape: (batch_size)

             pred_scores.extend(scores.cpu().numpy())
             # Map class indices back to names
             pred_classes.extend([idx_to_name_map.get(c.item(), "Unknown") for c in classes])

    print("Example Predicted Scores:", [f"{s:.3f}" for s in pred_scores[:10]])
    print("Example Mapped Classes:", pred_classes[:10])
    return pred_scores, pred_classes
# ================================================

# ================================================
# === ONNX Export Function ===
# ================================================
def export_to_onnx(model, dummy_input, save_path_onnx, device):
    """ Exports the PyTorch model to ONNX format. """
    print(f"\n--- Exporting Model to ONNX ({save_path_onnx}) ---")
    model.eval() # Ensure model is in evaluation mode
    model.to(device) # Ensure model is on the correct device
    dummy_input = dummy_input.to(device) # Ensure dummy input is on the correct device

    try:
        # Export the model
        torch.onnx.export(model,                     # model being run
                          dummy_input,               # model input (or a tuple for multiple inputs)
                          save_path_onnx,            # where to save the model (can be a file or file-like object)
                          export_params=True,        # store the trained parameter weights inside the model file
                          opset_version=11,          # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names = ['input'],   # the model's input names
                          output_names = ['output'], # the model's output names
                          dynamic_axes={'input' : {0 : 'batch_size', 1: 'sequence_length'}, # variable length axes
                                        'output' : {0 : 'batch_size'}})
        print("Model successfully exported to ONNX.")

        # Verify the ONNX model
        print("Verifying ONNX model...")
        onnx_model = onnx.load(save_path_onnx)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification successful.")

        # Optional: Test inference with ONNX Runtime
        # ort_session = onnxruntime.InferenceSession(save_path_onnx)
        # ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        # ort_outs = ort_session.run(None, ort_inputs)
        # print("ONNX Runtime inference test successful.")
        # print("  Input shape:", dummy_input.shape)
        # print("  Output shape (ONNX):", ort_outs[0].shape)

        return True

    except Exception as e:
        print(f"\n!!! ERROR during ONNX export or verification: {e} !!!")
        traceback.print_exc()
        # Clean up potentially corrupted file
        if os.path.exists(save_path_onnx):
            try: os.remove(save_path_onnx)
            except OSError: pass
        return False
# ================================================


# ================================================
# === Main Execution ===
# ================================================
if __name__ == "__main__":
    overall_start_time = time.time()
    print("--- Starting Engagement Prediction Script ---")

    # --- Create Save Directory ---
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Load Data ---
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = None, None, None
    # Define variables for ONNX export shape outside the try block
    SEQ_LEN, NUM_LANDMARKS, NUM_COORDS = None, None, None
    try:
        # Assuming get_dataloader determines sequence length from config or data
        train_loader = get_dataloader(CONFIG_PATH, 'Train', batch_size_override=BATCH_SIZE)
        val_loader = get_dataloader(CONFIG_PATH, 'Validation', batch_size_override=BATCH_SIZE)
        test_loader = get_dataloader(CONFIG_PATH, 'Test', batch_size_override=BATCH_SIZE)
        if not train_loader or not val_loader or not test_loader:
            raise ValueError("One or more dataloaders failed to initialize.")
        # Infer sequence length from a sample batch if possible (needed for ONNX dummy input)
        # This assumes get_dataloader yields tensors of shape (batch, seq, landmarks, coords)
        try:
            sample_inputs, _ = next(iter(train_loader))
            if isinstance(sample_inputs, torch.Tensor) and sample_inputs.ndim == 4:
                 SEQ_LEN = sample_inputs.shape[1]
                 NUM_LANDMARKS = sample_inputs.shape[2]
                 NUM_COORDS = sample_inputs.shape[3]
                 # Recalculate INPUT_DIM based on actual data? Or trust config? For now, use data shape.
                 ACTUAL_INPUT_DIM = NUM_LANDMARKS * NUM_COORDS
                 print(f"Inferred from data: Seq Len={SEQ_LEN}, Input Dim={ACTUAL_INPUT_DIM}")
                 # Optional: Check if ACTUAL_INPUT_DIM matches configured INPUT_DIM
                 if ACTUAL_INPUT_DIM != INPUT_DIM:
                     print(f"Warning: Inferred input dim ({ACTUAL_INPUT_DIM}) differs from configured INPUT_DIM ({INPUT_DIM}). Using configured value for model.")
                     # Keep INPUT_DIM as configured for model, but use inferred shape for dummy input
            else:
                raise ValueError("Could not infer input shape from dataloader (expected 4D tensor).")
        except StopIteration:
             raise ValueError("Training dataloader is empty.")
        except Exception as e_infer:
             print(f"Warning: Could not infer input shape from data ({e_infer}). Check dataloader output.")
             # Attempt to use placeholder values ONLY if inference failed
             if SEQ_LEN is None or NUM_LANDMARKS is None or NUM_COORDS is None:
                 print("Attempting to use placeholder values for ONNX export shape.")
                 SEQ_LEN = 30 # Example placeholder sequence length
                 # Need placeholders for landmarks/coords based on INPUT_DIM if possible
                 # This part is tricky without knowing the structure. Assuming 3 coords:
                 if INPUT_DIM % 3 == 0:
                     NUM_LANDMARKS = INPUT_DIM // 3
                     NUM_COORDS = 3
                     print(f"Using placeholders: Seq Len={SEQ_LEN}, Landmarks={NUM_LANDMARKS}, Coords={NUM_COORDS}")
                 else:
                      raise ValueError("Cannot determine placeholder shape for ONNX export from INPUT_DIM.")


    except Exception as e:
        print(f"\n!!! ERROR during DataLoader creation or shape inference: {e} !!!")
        traceback.print_exc()
        exit()
    print("Datasets loaded successfully.")

    # --- Initialize Model ---
    model = None
    print("\nInitializing model...")
    try:
        # Instantiate model definition using configured INPUT_DIM
        model_instance = EngagementRegressionModel(
            input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=1,
            num_layers=NUM_GRU_LAYERS, dropout=DROPOUT_RATE, bidirectional=True
        ).to(DEVICE)

        # Attempt to load saved state if requested
        if LOAD_SAVED_STATE:
            if os.path.exists(MODEL_SAVE_PATH_PTH):
                print(f"Attempting to load saved state from: {MODEL_SAVE_PATH_PTH}")
                try:
                    model_instance.load_state_dict(torch.load(MODEL_SAVE_PATH_PTH, map_location=DEVICE))
                    model = model_instance # Assign loaded model
                    print("Model state loaded successfully.")
                except Exception as e:
                    print(f"Warning: Failed to load state dict from {MODEL_SAVE_PATH_PTH}: {e}")
                    print("Proceeding with newly initialized model.")
            else:
                print(f"Warning: Saved state file not found at {MODEL_SAVE_PATH_PTH}.")
                print("Proceeding with newly initialized model.")

        # If model wasn't loaded, use the fresh instance
        if model is None:
            print("Using newly initialized model.")
            model = model_instance

        # --- Initialize Optimizer and Loss ---
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        print("\nModel Summary:")
        print(model)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {num_params:,}")

    except Exception as e:
        print(f"\n!!! ERROR during model initialization or loading: {e} !!!")
        traceback.print_exc()
        exit()

    # --- Train ---
    trained_model = None
    history = None
    # Only train if we didn't load a pre-trained model or if explicitly forced
    # (Currently, LOAD_SAVED_STATE just loads it, doesn't skip training)
    # Add a flag like SKIP_TRAINING_IF_LOADED if needed.
    print("\nStarting model training process...")
    try:
        trained_model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS,
            DEVICE, MODEL_SAVE_PATH_PTH, SAVE_BEST_MODEL_PTH
        )
    except KeyboardInterrupt:
         print("\n--- Training interrupted by user ---")
         # Decide if you want to proceed with the model state at interruption
         # For now, we'll use the 'model' variable which holds the last state
         trained_model = model # Use the model state at interruption
         history = None # History might be incomplete
         print("Proceeding with model state at interruption (best saved state might be loaded if available).")
    except Exception as e:
        print(f"\n!!! ERROR during training: {e} !!!")
        traceback.print_exc()
        # Attempt to load the best saved model if training failed mid-way
        if SAVE_BEST_MODEL_PTH and os.path.exists(MODEL_SAVE_PATH_PTH):
            print("Attempting to load best saved model due to training error...")
            try:
                model.load_state_dict(torch.load(MODEL_SAVE_PATH_PTH, map_location=DEVICE))
                trained_model = model
                print("Successfully loaded best saved model.")
            except Exception as le:
                 print(f"Could not load best saved model: {le}. No trained model available.")
                 trained_model = None
        else:
            trained_model = None # No trained model available
        history = None # History is likely invalid


    # --- Plot Training History ---
    if history:
        plot_training_history(history, save_dir=SAVE_DIR)

    # --- Evaluate ---
    if trained_model and test_loader:
        try:
            evaluate_model(trained_model, test_loader, criterion, DEVICE, IDX_TO_NAME_MAP)
        except Exception as e:
            print(f"\n!!! ERROR during evaluation: {e} !!!")
            traceback.print_exc()
    elif not trained_model:
        print("\nSkipping evaluation: No valid trained model available.")
    else:
         print("\nSkipping evaluation: Test loader not available.")


    # --- Export to ONNX ---
    if trained_model and SAVE_FINAL_MODEL_ONNX:
        # Check if shape variables were successfully determined
        if SEQ_LEN is not None and NUM_LANDMARKS is not None and NUM_COORDS is not None:
            try:
                # Create a dummy input tensor matching the model's expected input shape
                # Use batch_size=1 for standard export
                # Shape: (batch_size, seq_len, num_landmarks, coords)
                dummy_input = torch.randn(1, SEQ_LEN, NUM_LANDMARKS, NUM_COORDS, device='cpu') # Create on CPU first
                export_to_onnx(trained_model, dummy_input, MODEL_SAVE_PATH_ONNX, DEVICE)
            except Exception as e:
                print(f"\n!!! ERROR during ONNX export preparation or execution: {e} !!!")
                traceback.print_exc()
        else:
             print("\n!!! ERROR during ONNX export: Could not determine input shape (SEQ_LEN, NUM_LANDMARKS, NUM_COORDS). Export skipped. !!!")
             print("   Check dataloader and shape inference steps.")

    elif not trained_model:
         print("\nSkipping ONNX export: No valid trained model.")
    elif not SAVE_FINAL_MODEL_ONNX:
        print("\nSkipping ONNX export: Saving disabled in configuration.")

    overall_end_time = time.time()
    print("\n--- Script Finished ---")
    print(f"Total script execution time: {(overall_end_time - overall_start_time):.2f}s")
# ================================================
