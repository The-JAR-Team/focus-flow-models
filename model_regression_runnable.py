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
import copy # For saving best model state in memory

# --- Plotting and Metrics ---
try:
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, r2_score
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    SKLEARN_AVAILABLE = True
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn, numpy, seaborn, or matplotlib not found.")
    print("         Classification metrics and plotting will be unavailable.")
    print("         Install using: pip install scikit-learn numpy seaborn matplotlib pandas")
    SKLEARN_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = False # Add flag for plotting

# --- Import DataLoader function ---
try:
    from Preprocess.Pipeline.InspectData import get_dataloader
except ImportError:
    print("ERROR: Could not import get_dataloader from Preprocess.Pipeline.InspectData.")
    print("       Please ensure inspect.py containing get_dataloader is accessible.")
    exit()

# ================================================
# === Configuration ===
# ================================================
CONFIG_PATH = "./Preprocess/Pipeline/Pipelines/configs/ENGAGENET_10fps_quality95_randdist.json"
MODEL_SAVE_PATH = "./engagement_model_regression_0_1.pth"
SAVE_BEST_MODEL = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 50
HIDDEN_DIM = 256
NUM_GRU_LAYERS = 2
DROPOUT_RATE = 0.4
# --------------------------

print(f"--- Configuration ---")
print(f"Device: {DEVICE}")
print(f"Config Path: {CONFIG_PATH}")
print(f"Save Best Model: {SAVE_BEST_MODEL}")
if SAVE_BEST_MODEL:
    print(f"Model Save Path: {MODEL_SAVE_PATH}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Target Score Range: [0.0, 1.0]")
print(f"-------------------")

# ================================================
# === Mappings ===
# ================================================
# (Mappings remain the same as the previous regression version)
LABEL_TO_IDX_MAP = {
    'Not Engaged': 0, 'Barely Engaged': 1, 'Engaged': 2, 'Highly Engaged': 3,
    'not engaged': 0, 'not-engaged': 0, 'Not-Engaged': 0,
    'barely engaged': 1, 'barely-engaged': 1, 'Barely-engaged': 1,
    'highly engaged': 3, 'highly-engaged': 3, 'Highly-Engaged': 3,
    'snp(subject not present)': 4, 'SNP(Subject Not Present)': 4, 'SNP': 4,
}
IDX_TO_SCORE_MAP = {4: 0.0, 0: 0.25, 1: 0.5, 2: 0.75, 3: 1.0}
IDX_TO_NAME_MAP = {0: 'Not Engaged', 1: 'Barely Engaged', 2: 'Engaged', 3: 'Highly Engaged', 4: 'SNP'}
print("Label -> Index -> Score Mapping Initialized.")
# ================================================


# ================================================
# === Model Definition ===
# ================================================
# (Model definition remains the same)
class EngagementRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.5, bidirectional=True):
        super().__init__()
        self.input_dim = input_dim; self.hidden_dim = hidden_dim; self.num_layers = num_layers
        self.bidirectional = bidirectional; self.num_classes = output_dim
        self.frame_norm = nn.LayerNorm(input_dim); gru_input_dim = input_dim
        self.gru = nn.GRU(gru_input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout); self.fc1 = nn.Linear(gru_output_dim, hidden_dim // 2)
        self.relu = nn.ReLU(); self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.output_activation = nn.Sigmoid()
    def forward(self, x):
        batch_size, seq_len, num_landmarks, coords = x.shape; x = x.reshape(batch_size, seq_len, -1)
        if x.shape[2] != self.input_dim: raise ValueError(f"Input dim mismatch")
        x = self.frame_norm(x); gru_out, hn = self.gru(x)
        if self.bidirectional: last_hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        else: last_hidden = hn[-1,:,:]
        out = self.dropout(last_hidden); out = self.fc1(out); out = self.relu(out); out = self.fc2(out)
        out = self.output_activation(out); return out
# ================================================


# ================================================
# === Utility Functions ===
# ================================================
# (get_targets and map_score_to_class_idx remain the same)
def get_targets(label_batch: Dict[str, List[Any]], label_to_idx_map: Dict[str, int], idx_to_score_map: Dict[int, float]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if 'engagement_string' not in label_batch: print("Warn: 'engagement_string' missing!"); return None
    string_labels = label_batch['engagement_string']; target_scores = []; target_indices = []
    valid_batch = True
    for lbl in string_labels:
        processed_lbl = str(lbl).strip()
        class_idx = label_to_idx_map.get(processed_lbl)
        if class_idx is None: print(f"Warn: Unknown label: '{lbl}'. Skip batch."); valid_batch = False; break
        else:
            score = idx_to_score_map.get(class_idx)
            if score is None: print(f"Warn: No score for index {class_idx}. Skip batch."); valid_batch = False; break
            target_indices.append(class_idx); target_scores.append(score)
    if not valid_batch: return None
    scores_tensor = torch.tensor(target_scores, dtype=torch.float).unsqueeze(1)
    indices_tensor = torch.tensor(target_indices, dtype=torch.long)
    return scores_tensor, indices_tensor

def map_score_to_class_idx(score_tensor: torch.Tensor) -> torch.Tensor:
    thresholds = [0.125, 0.375, 0.625, 0.875]; score_tensor_flat = score_tensor.squeeze()
    preds = torch.full_like(score_tensor_flat, -1, dtype=torch.long)
    preds = torch.where(score_tensor_flat < thresholds[0], 4, preds); preds = torch.where((score_tensor_flat >= thresholds[0]) & (score_tensor_flat < thresholds[1]), 0, preds)
    preds = torch.where((score_tensor_flat >= thresholds[1]) & (score_tensor_flat < thresholds[2]), 1, preds); preds = torch.where((score_tensor_flat >= thresholds[2]) & (score_tensor_flat < thresholds[3]), 2, preds)
    preds = torch.where(score_tensor_flat >= thresholds[3], 3, preds)
    if score_tensor.dim() > 1 and score_tensor.shape[-1] == 1 : preds = preds.unsqueeze(-1)
    return preds
# ================================================


# ================================================
# === Training Function (Modified Print Order) ===
# ================================================
def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device,
    save_path: str, save_model: bool
    ) -> Tuple[nn.Module, Dict]: # Return model and history dictionary
    """ Trains the regression model, saving best based on validation loss, returns history. """
    best_val_loss = float('inf')
    best_model_state_dict = None
    final_model = model
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy_mapped': []}

    print("\n--- Starting Training (Regression [0, 1]) ---")
    if save_model:
        print(f"Best model will be saved to: {save_path}")
    else:
        print("Model saving is disabled.")

    # --- Epoch Loop ---
    for epoch in range(num_epochs):
        # Print epoch number clearly BEFORE the progress bars for that epoch
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        total_samples_train = 0 # Track total samples for loss average
        # Using tqdm as an iterator handles closing automatically
        train_pbar = tqdm(train_loader, desc="  Training", leave=False, ncols=100)
        for inputs, labels_dict in train_pbar:
            if not isinstance(inputs, torch.Tensor): continue
            inputs = inputs.to(device)
            targets = get_targets(labels_dict, LABEL_TO_IDX_MAP, IDX_TO_SCORE_MAP)
            if targets is None: continue
            target_scores, _ = targets
            target_scores = target_scores.to(device)

            batch_size = inputs.size(0)
            total_samples_train += batch_size

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target_scores)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size
            train_pbar.set_postfix(loss=loss.item())
        # train_pbar closes automatically here

        epoch_train_loss = running_loss / total_samples_train if total_samples_train > 0 else 0
        history['train_loss'].append(epoch_train_loss) # Store train loss

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects_mapped = 0
        total_val_samples = 0 # Track total samples for loss average AND valid samples for accuracy
        total_val_valid_samples_acc = 0 # Count only non-ignored samples for mapped acc

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="  Validation", leave=False, ncols=100)
            for inputs, labels_dict in val_pbar:
                if not isinstance(inputs, torch.Tensor): continue
                inputs = inputs.to(device)
                targets = get_targets(labels_dict, LABEL_TO_IDX_MAP, IDX_TO_SCORE_MAP)
                if targets is None: continue
                target_scores, target_indices = targets # Need indices for mapped accuracy
                target_scores = target_scores.to(device)
                target_indices = target_indices.to(device) # Keep on device

                batch_size = inputs.size(0)
                total_val_samples += batch_size

                pred_scores = model(inputs)
                loss = criterion(pred_scores, target_scores)
                val_loss += loss.item() * batch_size

                # Calculate Mapped Accuracy
                pred_classes = map_score_to_class_idx(pred_scores)
                # Compare against original class indices
                # Note: We are calculating accuracy on ALL validation samples here, including SNP
                # If you wanted accuracy only on engagement classes 0-3, you would add a mask here.
                val_corrects_mapped += torch.sum(pred_classes.squeeze() == target_indices)
                total_val_valid_samples_acc += batch_size # Assuming accuracy calculated over all classes 0-4

        # val_pbar closes automatically here

        epoch_val_loss = val_loss / total_val_samples if total_val_samples > 0 else float('inf')
        epoch_val_acc_mapped = val_corrects_mapped.double() / total_val_valid_samples_acc if total_val_valid_samples_acc > 0 else 0.0
        history['val_loss'].append(epoch_val_loss) # Store val loss
        history['val_accuracy_mapped'].append(epoch_val_acc_mapped.item()) # Store val mapped accuracy

        # --- Print Epoch Summary AFTER validation is done ---
        print(f"  Epoch {epoch+1} Summary:")
        print(f"    Train MSE Loss: {epoch_train_loss:.6f}") # More precision for loss
        print(f"    Val MSE Loss:   {epoch_val_loss:.6f}")
        print(f"    Val Acc (map):  {epoch_val_acc_mapped:.4f}")
        # ---------------------------------------------------

        # --- Save Check ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            print(f"    * Best Val Loss so far: {best_val_loss:.6f}. ", end="") # Combine print
            if save_model:
                print(f"Saving model to {save_path}...")
                try:
                    torch.save(model.state_dict(), save_path)
                except Exception as e:
                    print(f"\n      ERROR saving model: {e}")
            else:
                print("(Saving disabled, keeping state in memory)")
                best_model_state_dict = copy.deepcopy(model.state_dict())

    # --- End of Epoch Loop ---

    print('\n--- Training Finished ---')
    print(f"Best Validation MSE Loss achieved: {best_val_loss:.6f}")

    # --- Load Best State ---
    # (Loading logic remains the same)
    if save_model:
        if os.path.exists(save_path):
            print(f"Loading best model weights from {save_path}")
            try: model.load_state_dict(torch.load(save_path, map_location=device)); final_model = model
            except Exception as e: print(f"Error loading saved state: {e}.")
        else: print("Warning: Best model file not found.")
    elif best_model_state_dict is not None:
        print("Loading best model state from memory."); model.load_state_dict(best_model_state_dict); final_model = model
    else: print("Returning model state from the last epoch.")

    return final_model, history # Return model AND history
# ================================================


# ================================================
# === Plotting Function ===
# ================================================
# (Plotting function remains the same)
def plot_training_history(history: Dict, save_dir: str = "."):
    if not MATPLOTLIB_AVAILABLE: print("Plotting skipped: matplotlib not available."); return
    epochs = range(1, len(history['train_loss']) + 1)
    plt.style.use('seaborn-v0_8-darkgrid')
    try: # Loss Plot
        plt.figure(figsize=(12, 5)); plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss (MSE)'); plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss (MSE)')
        plt.title('Training and Validation Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss (MSE)'); plt.legend(); plt.grid(True)
        plt.subplot(1, 2, 2) # Loss Difference Plot
        loss_diff = [v - t for v, t in zip(history['val_loss'], history['train_loss'])]
        plt.plot(epochs, loss_diff, 'go-', label='Validation - Training Loss'); plt.title('Loss Difference (Val - Train)')
        plt.xlabel('Epochs'); plt.ylabel('Loss Difference'); plt.axhline(0, color='grey', lw=0.5, linestyle='--'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plot_path = os.path.join(save_dir, "loss_curves.png"); plt.savefig(plot_path)
        print(f"Loss curves plot saved to {plot_path}"); plt.close()
    except Exception as e: print(f"Error plotting loss curves: {e}"); plt.close()
    if 'val_accuracy_mapped' in history and history['val_accuracy_mapped']: # Accuracy Plot
        try:
            plt.figure(figsize=(7, 5))
            plt.plot(epochs, history['val_accuracy_mapped'], 'mo-', label='Validation Accuracy (Mapped)')
            plt.title('Validation Accuracy (Mapped from Regression)'); plt.xlabel('Epochs'); plt.ylabel('Accuracy')
            plt.ylim(0, 1); plt.legend(); plt.grid(True); plt.tight_layout()
            plot_path = os.path.join(save_dir, "mapped_accuracy_curve.png"); plt.savefig(plot_path)
            print(f"Mapped accuracy curve plot saved to {plot_path}"); plt.close()
        except Exception as e: print(f"Error plotting mapped accuracy curve: {e}"); plt.close()
# ================================================


# ================================================
# === Evaluation Function ===
# ================================================
# (Evaluation function remains the same)
def evaluate_model(model, test_loader, criterion, device):
    print("\n--- Evaluating on Test Set (Regression + Mapped Classification) ---")
    model.eval(); test_loss = 0.0; test_samples = 0
    all_pred_scores = []; all_target_scores = []; all_pred_classes = []; all_target_classes = []
    final_cls_acc = 0.0
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing", leave=False, ncols=100)
        for inputs, labels_dict in test_pbar:
            if not isinstance(inputs, torch.Tensor): continue
            inputs = inputs.to(device); targets = get_targets(labels_dict, LABEL_TO_IDX_MAP, IDX_TO_SCORE_MAP)
            if targets is None: continue
            target_scores, target_indices = targets; target_scores = target_scores.to(device); target_indices = target_indices.to(device)
            batch_size = inputs.size(0); test_samples += batch_size
            pred_scores = model(inputs); loss = criterion(pred_scores, target_scores)
            test_loss += loss.item() * batch_size
            pred_classes = map_score_to_class_idx(pred_scores)
            all_pred_scores.extend(pred_scores.squeeze().cpu().numpy()); all_target_scores.extend(target_scores.squeeze().cpu().numpy())
            all_pred_classes.extend(pred_classes.squeeze().cpu().numpy()); all_target_classes.extend(target_indices.cpu().numpy())
    final_test_loss = test_loss / test_samples if test_samples > 0 else 0
    print(f"\nTest MSE Loss: {final_test_loss:.6f}") # More precision for loss
    if SKLEARN_AVAILABLE and all_target_classes and all_pred_classes:
        final_cls_acc = accuracy_score(all_target_classes, all_pred_classes)
        print(f"Test Classification Accuracy (mapped): {final_cls_acc:.4f}")
        print("\nClassification Report (mapped):"); report_labels = sorted(IDX_TO_NAME_MAP.keys()); target_names = [IDX_TO_NAME_MAP[i] for i in report_labels]
        print(classification_report(all_target_classes, all_pred_classes, labels=report_labels, target_names=target_names, digits=4, zero_division=0))
        print("Confusion Matrix (mapped):"); cm = confusion_matrix(all_target_classes, all_pred_classes, labels=report_labels); print(cm)
        try:
            plt.figure(figsize=(9, 7)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
            plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Confusion Matrix (Mapped from Regression)'); plt.tight_layout()
            plt.savefig("confusion_matrix_regression_mapped.png"); print("Confusion matrix plot saved."); plt.close()
        except Exception as e: print(f"Could not plot confusion matrix: {e}"); plt.close()
    elif not SKLEARN_AVAILABLE: print("\nSkipping Classification Report/Matrix: scikit-learn not installed.")
    else: print("\nCould not generate classification metrics.")
    if SKLEARN_AVAILABLE and all_target_scores and all_pred_scores:
        mae = mean_absolute_error(all_target_scores, all_pred_scores); r2 = r2_score(all_target_scores, all_pred_scores)
        print(f"\nTest MAE: {mae:.4f}"); print(f"Test R^2 Score: {r2:.4f}")
    elif not SKLEARN_AVAILABLE: print("\nSkipping Regression Metrics: scikit-learn not installed.")
    return final_test_loss, final_cls_acc
# ================================================


# ================================================
# === Prediction Function ===
# ================================================
# (Prediction function remains the same)
def predict_engagement_score(model, data_loader, device):
    model.eval(); pred_scores = []; pred_classes = []
    print("\n--- Making Predictions (Regression Scores + Mapped Classes) ---")
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Predicting", leave=False, ncols=100)
        for inputs, _ in pbar:
             if not isinstance(inputs, torch.Tensor): continue
             inputs = inputs.to(device); outputs = model(inputs)
             scores = outputs.squeeze(-1); classes = map_score_to_class_idx(scores)
             pred_scores.extend(scores.cpu().numpy())
             pred_classes.extend([IDX_TO_NAME_MAP.get(c.item(), "Unknown") for c in classes])
    print("Example Predicted Scores:", [f"{s:.3f}" for s in pred_scores[:10]])
    print("Example Mapped Classes:", pred_classes[:10])
    return pred_scores, pred_classes
# ================================================


# ================================================
# === Main Execution ===
# ================================================
if __name__ == "__main__":
    start_overall_time = time.time()
    print("--- Starting Engagement Prediction Training Script (Regression [0, 1]) ---")
    print(f"Model saving enabled: {SAVE_BEST_MODEL}")

    # --- Load Data ---
    print("\nLoading datasets...")
    try:
        train_loader = get_dataloader(CONFIG_PATH, 'Train', batch_size_override=BATCH_SIZE)
        val_loader = get_dataloader(CONFIG_PATH, 'Validation', batch_size_override=BATCH_SIZE)
        test_loader = get_dataloader(CONFIG_PATH, 'Test', batch_size_override=BATCH_SIZE)
        if not train_loader or not val_loader or not test_loader: print("\nERROR: Failed dataloader load. Exiting."); exit()
    except Exception as e: print(f"\n!!! ERROR DataLoader creation: {e} !!!"); traceback.print_exc(); exit()
    print("Datasets loaded successfully.")

    # --- Initialize Model, Loss, Optimizer ---
    print("\nInitializing regression model...")
    try:
        INPUT_DIM = 478 * 3
        model = EngagementRegressionModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=1,
                                          num_layers=NUM_GRU_LAYERS, dropout=DROPOUT_RATE, bidirectional=True).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        print("Model Summary:"); print(model); num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {num_params:,}")
    except Exception as e: print(f"\n!!! ERROR model initialization: {e} !!!"); traceback.print_exc(); exit()

    # --- Train ---
    trained_model = None
    history = None
    try:
        # Capture history dictionary
        trained_model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS,
            DEVICE, MODEL_SAVE_PATH, SAVE_BEST_MODEL
        )
    except Exception as e: print(f"\n!!! ERROR during training: {e} !!!"); traceback.print_exc(); exit()

    # --- Plot Training History ---
    if history:
        plot_training_history(history, save_dir=".") # Pass the history
    # ---------------------------

    # --- Evaluate ---
    if trained_model:
        try:
            evaluate_model(trained_model, test_loader, criterion, DEVICE)
        except Exception as e: print(f"\n!!! ERROR during evaluation: {e} !!!"); traceback.print_exc()
    else: print("Training did not return valid model. Skipping evaluation.")

    # --- Predict (Optional Example) ---
    # if trained_model and test_loader: try: predict_engagement_score(trained_model, test_loader, DEVICE)
    # except Exception as e: print(f"\n!!! ERROR prediction: {e} !!!"); traceback.print_exc()

    end_overall_time = time.time()
    print("\n--- Script Finished ---")
    print(f"Total script execution time: {(end_overall_time - start_overall_time):.2f}s")
# ================================================
