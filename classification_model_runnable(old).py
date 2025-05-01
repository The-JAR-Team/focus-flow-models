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
# --- For evaluation ---
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy

# --- Import DataLoader function ---
try:
    from Preprocess.Pipeline.InspectData import get_dataloader
except ImportError:
    print("ERROR: Could not import get_dataloader from Preprocess.Pipeline.InspectData.")
    exit()

# === Configuration ===
CONFIG_PATH = "./Preprocess/Pipeline/Pipelines/configs/ENGAGENET_10fps_quality95_randdist.json"
MODEL_SAVE_PATH = "./engagement_model_gru_5class.pth" # Changed save path slightly
SAVE_BEST_MODEL = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 25
HIDDEN_DIM = 256
NUM_GRU_LAYERS = 2
DROPOUT_RATE = 0.4
# --- Updated number of classes ---
NUM_CLASSES = 5  # Now 5 classes: 0-3 for engagement, 4 for SNP
# ---------------------------------
# IGNORE_INDEX = -100 # No longer needed for SNP

# --- Updated Label Mapping ---
LABEL_MAP = {
    # Standard / Target Keys
    'Not Engaged': 0,
    'Barely Engaged': 1,
    'Engaged': 2,
    'Highly Engaged': 3,
    # Observed Variations (mapping to same indices)
    'not engaged': 0, 'not-engaged': 0, 'Not-Engaged': 0,
    'barely engaged': 1, 'barely-engaged': 1, 'Barely-engaged': 1,
    'highly engaged': 3, 'highly-engaged': 3, 'Highly-Engaged': 3,
    # Special Cases (Now mapped to class 4)
    'snp(subject not present)': 4,
    'SNP(Subject Not Present)': 4,
    'SNP': 4,
}
# Create reverse map for interpreting predictions (now includes class 4)
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items() if '-' not in k and ' ' not in k and '(' not in k} # Create cleaner names for reporting
# Manually ensure all target indices are present and have a reasonable name
IDX_TO_LABEL[0] = 'Not Engaged'
IDX_TO_LABEL[1] = 'Barely Engaged'
IDX_TO_LABEL[2] = 'Engaged'
IDX_TO_LABEL[3] = 'Highly Engaged'
IDX_TO_LABEL[4] = 'SNP' # Use short name for reporting
print("Label Map Initialized (5 Classes):")
# for k, v in LABEL_MAP.items(): print(f"  '{k}' -> {v}")
print("Index to Label for Reporting:")
for i in sorted(IDX_TO_LABEL.keys()): print(f"  {i} -> '{IDX_TO_LABEL[i]}'")
# ---------------------------

# === Model Definition ===
class EngagementGRUModel(nn.Module):
    # (Model definition remains IDENTICAL - relies on NUM_CLASSES constant)
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5, bidirectional=True):
        super().__init__()
        self.input_dim = input_dim; self.hidden_dim = hidden_dim; self.num_layers = num_layers
        self.bidirectional = bidirectional; self.num_classes = num_classes # Now uses 5
        self.frame_norm = nn.LayerNorm(input_dim); gru_input_dim = input_dim
        self.gru = nn.GRU(gru_input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout); self.fc1 = nn.Linear(gru_output_dim, hidden_dim // 2)
        self.relu = nn.ReLU(); self.fc2 = nn.Linear(hidden_dim // 2, num_classes) # Output layer size is now 5
    def forward(self, x):
        batch_size, seq_len, num_landmarks, coords = x.shape; x = x.reshape(batch_size, seq_len, -1)
        if x.shape[2] != self.input_dim: raise ValueError(f"Input dim mismatch")
        x = self.frame_norm(x); gru_out, hn = self.gru(x)
        if self.bidirectional: last_hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        else: last_hidden = hn[-1,:,:]
        out = self.dropout(last_hidden); out = self.fc1(out); out = self.relu(out); out = self.fc2(out)
        return out

# === Utility Functions ===
def map_labels(label_batch: Dict[str, List[Any]], label_map: Dict[str, int]) -> Optional[torch.Tensor]:
    """Maps string labels to integer indices."""
    # Logic remains the same, but now maps SNP to 4 instead of -100
    if 'engagement_string' not in label_batch:
        print("Critical Warning: 'engagement_string' key missing!"); return None
    string_labels = label_batch['engagement_string']; mapped_labels = []
    valid_batch = True
    for lbl in string_labels:
        processed_lbl = str(lbl).strip()
        idx = label_map.get(processed_lbl) # Check against the expanded map
        if idx is None:
            # Still warn for truly unknown labels
            print(f"Warning: Unknown label encountered (will skip batch): '{lbl}' (Processed: '{processed_lbl}')")
            valid_batch = False; break
        else:
            mapped_labels.append(idx)
    if not valid_batch: return None
    return torch.tensor(mapped_labels, dtype=torch.long)

# === Modified Training Function (Accuracy calculation reverted) ===
def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device,
    save_path: str, save_model: bool
    ):
    """Trains the model, handles 5 classes, optionally saves."""
    best_val_acc = 0.0
    best_model_state_dict = None
    print("\n--- Starting Training (5 Classes) ---")
    if save_model: print(f"Best model will be saved to: {save_path}")
    else: print("Model saving is disabled.")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}\n{'-' * 10}")

        # --- Training Phase ---
        model.train()
        running_loss = 0.0; running_corrects = 0; total_samples = 0 # Count all samples for acc now
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels_dict in train_pbar:
            if not isinstance(inputs, torch.Tensor): continue
            inputs = inputs.to(device)
            targets = map_labels(labels_dict, LABEL_MAP)
            if targets is None: continue # Skip if unknown label found
            targets = targets.to(device)

            batch_size = inputs.size(0); total_samples += batch_size # Count all samples
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets) # Loss calculated over all 5 classes
            _, preds = torch.max(outputs, 1)
            loss.backward(); optimizer.step()

            # --- Accuracy Calculation (now includes class 4) ---
            running_corrects += torch.sum(preds == targets.data)
            batch_acc = torch.sum(preds == targets.data).item() / batch_size if batch_size > 0 else 0
            # ----------------------------------------------------

            running_loss += loss.item() * batch_size
            train_pbar.set_postfix(loss=loss.item(), batch_acc=batch_acc)

        epoch_train_loss = running_loss / total_samples if total_samples > 0 else 0 # Avg loss over all samples
        epoch_train_acc = running_corrects.double() / total_samples if total_samples > 0 else 0.0 # Avg acc over all samples
        print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0; val_corrects = 0; total_val_samples = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", leave=False)
            for inputs, labels_dict in val_pbar:
                if not isinstance(inputs, torch.Tensor): continue
                inputs = inputs.to(device)
                targets = map_labels(labels_dict, LABEL_MAP)
                if targets is None: continue
                targets = targets.to(device)

                batch_size = inputs.size(0); total_val_samples += batch_size # Count all
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * batch_size

                # --- Accuracy Calculation (now includes class 4) ---
                val_corrects += torch.sum(preds == targets.data)
                # ----------------------------------------------------

        epoch_val_loss = val_loss / total_val_samples if total_val_samples > 0 else 0
        epoch_val_acc = val_corrects.double() / total_val_samples if total_val_samples > 0 else 0.0
        print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        # --- Save Check ---
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            print(f" * New best validation accuracy: {best_val_acc:.4f}")
            if save_model:
                print(f"   Saving best model state to {save_path}")
                try: torch.save(model.state_dict(), save_path)
                except Exception as e: print(f"   ERROR saving model: {e}")
            else:
                print("   (Keeping best model state in memory)")
                best_model_state_dict = copy.deepcopy(model.state_dict())

    print('\n--- Training Finished ---')
    print(f"Best Validation Accuracy achieved: {best_val_acc:.4f}")

    # --- Load Best State ---
    # (Loading logic remains the same)
    if save_model:
        if os.path.exists(save_path):
            print(f"Loading best model weights from {save_path}")
            try: model.load_state_dict(torch.load(save_path, map_location=device))
            except Exception as e: print(f"Error loading saved model state: {e}.")
        else: print("Warning: Best model file not found.")
    elif best_model_state_dict is not None:
        print("Loading best model state from memory."); model.load_state_dict(best_model_state_dict)
    else: print("Returning model state from the last epoch.")

    return model

# === Modified Evaluation Function ===
def evaluate_model(model, test_loader, criterion, device):
    """Evaluates the 5-class model on the test set."""
    print("\n--- Evaluating on Test Set (5 Classes) ---")
    model.eval(); test_loss = 0.0; test_corrects = 0; test_samples = 0
    all_preds = []; all_targets = [] # Store all predictions and targets

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing", leave=False)
        for inputs, labels_dict in test_pbar:
            if not isinstance(inputs, torch.Tensor): continue
            inputs = inputs.to(device)
            targets = map_labels(labels_dict, LABEL_MAP) # Maps SNP to 4
            if targets is None: continue
            targets = targets.to(device)

            batch_size = inputs.size(0); test_samples += batch_size # Count all
            outputs = model(inputs)
            loss = criterion(outputs, targets) # Loss includes class 4
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * batch_size

            # --- Accuracy Calculation (includes class 4) ---
            test_corrects += torch.sum(preds == targets.data)
            # ---------------------------------------------
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    final_test_loss = test_loss / test_samples if test_samples > 0 else 0
    final_test_acc = test_corrects.double() / test_samples if test_samples > 0 else 0.0

    print(f"\nTest Loss: {final_test_loss:.4f}")
    print(f"Test Accuracy: {final_test_acc:.4f}")

    # --- Metrics Calculation (now includes class 4) ---
    if all_targets and all_preds:
        print("\nClassification Report:")
        # Ensure labels/target_names cover indices 0, 1, 2, 3, 4
        report_labels = sorted(IDX_TO_LABEL.keys()) # Should be [0, 1, 2, 3, 4]
        target_names = [IDX_TO_LABEL[i] for i in report_labels]
        print(classification_report(all_targets, all_preds, labels=report_labels, target_names=target_names, digits=4, zero_division=0))

        print("Confusion Matrix:")
        cm = confusion_matrix(all_targets, all_preds, labels=report_labels)
        print(cm)
        try: # Plotting
            plt.figure(figsize=(9, 7)) # Slightly larger figure for 5 classes
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
            plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Confusion Matrix (5 Classes)'); plt.tight_layout()
            plt.savefig("confusion_matrix_5class.png"); print("Confusion matrix plot saved to confusion_matrix_5class.png")
        except Exception as e: print(f"Could not plot confusion matrix: {e}")
    else:
        print("\nCould not generate metrics (no predictions/targets).")
    # -----------------------------------------------------

    return final_test_loss, final_test_acc


# === Prediction Function (Example) ===
def predict_engagement(model, data_loader, device):
    # (No change needed, IDX_TO_LABEL now includes SNP mapping)
    model.eval(); predictions = []
    print("\n--- Making Predictions (Example - 5 Classes) ---")
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Predicting", leave=False)
        for inputs, _ in pbar:
             if not isinstance(inputs, torch.Tensor): continue
             inputs = inputs.to(device); outputs = model(inputs); _, preds = torch.max(outputs, 1)
             predictions.extend([IDX_TO_LABEL.get(p.item(), f"Unknown Idx: {p.item()}") for p in preds])
    print("Example Predictions:", predictions[:10])
    return predictions


# === Main Script ===
if __name__ == "__main__":
    print("--- Starting Engagement Prediction Training Script (5 Classes) ---")
    print(f"Model saving enabled: {SAVE_BEST_MODEL}")

    # --- Load Data ---
    print("Loading datasets...")
    train_loader = get_dataloader(CONFIG_PATH, 'Train', batch_size_override=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(CONFIG_PATH, 'Validation', batch_size_override=BATCH_SIZE)
    test_loader = get_dataloader(CONFIG_PATH, 'Test', batch_size_override=BATCH_SIZE)
    if not train_loader or not val_loader or not test_loader: print("\nERROR: Failed to load dataloaders. Exiting."); exit()

    # --- Initialize Model, Loss, Optimizer ---
    print("Initializing model...")
    INPUT_DIM = 478 * 3
    # Model automatically adapts to NUM_CLASSES=5
    model = EngagementGRUModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
                               num_layers=NUM_GRU_LAYERS, dropout=DROPOUT_RATE, bidirectional=True).to(DEVICE)
    # --- Loss function without ignore_index ---
    criterion = nn.CrossEntropyLoss()
    # -----------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print("Model Summary:"); print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params:,}")

    # --- Train ---
    trained_model = None
    try:
        trained_model = train_model(
            model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS,
            DEVICE, MODEL_SAVE_PATH, SAVE_BEST_MODEL
            # No ignore_index needed here anymore
        )
    except Exception as e: print(f"\n!!! ERROR during training: {e} !!!"); traceback.print_exc(); exit()

    # --- Evaluate ---
    if trained_model:
        try:
            # Pass model trained on 5 classes
            evaluate_model(trained_model, test_loader, criterion, DEVICE)
        except Exception as e: print(f"\n!!! ERROR during evaluation: {e} !!!"); traceback.print_exc()
    else: print("Training did not return a valid model. Skipping evaluation.")

    print("\n--- Script Finished ---")
