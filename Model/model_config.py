# Model/model_config.py
import os
import torch
import torch.nn as nn # Import nn is now essential here

# ================================================
# === Configuration ===
# ================================================
CONFIG_PATH = "./Preprocess/Pipeline/Pipelines/configs/ENGAGENET_10fps_quality95_randdist.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data & Input/Output Dimensions ---
# These are still needed as they define the overall model interface
INPUT_DIM = 478 * 3 # Example: (num_landmarks * coordinates)
OUTPUT_DIM = 1      # Regression output (0 to 1)

# --- Training Hyperparameters ---
LEARNING_RATE = 0.00005
BATCH_SIZE = 32
NUM_EPOCHS = 40

# --- Model Architecture Definition (List of INSTANTIATED nn.Module objects) ---
# Hyperparameters are now defined *directly* within the layer instantiations.

MODEL_LAYERS = [
    # Note: The initial LayerNorm and reshape are handled by the model wrapper class.
    #       The list starts with the first layer *after* initial normalization.

    # 1. GRU Layer
    nn.GRU(
        input_size=INPUT_DIM,       # Use INPUT_DIM directly
        hidden_size=256,            # Define hidden_size here
        num_layers=2,               # Define num_layers here
        batch_first=True,
        dropout=0.4 if 2 > 1 else 0,# Define dropout here (conditional on num_layers)
        bidirectional=True          # Define bidirectional here
    ),

    # --- Layers applied to the *final hidden state* of the GRU ---
    # 2. Dropout after GRU
    nn.Dropout(0.4),                # Define dropout probability here

    # 3. First Fully Connected Layer
    nn.Linear(
        # Calculate in_features based on the GRU params defined above:
        # hidden_size * (2 if bidirectional else 1)
        in_features=256 * (2 if True else 1),
        out_features=128            # Define out_features here
    ),

    # 4. Activation
    nn.ReLU(),

    # 5. Second Fully Connected Layer (Output Layer)
    nn.Linear(
        in_features=128,            # Must match out_features of previous Linear layer
        out_features=OUTPUT_DIM     # Use OUTPUT_DIM directly
    ),

    # 6. Final Activation (for regression in [0, 1])
    nn.Sigmoid()
]

# --- Saving & Loading ---
MODEL_BASE_NAME = "engagement_model_instantiated_reg_0_2" # Updated version name
SAVE_DIR = "."
MODEL_SAVE_PATH_PTH = os.path.join(SAVE_DIR, f"{MODEL_BASE_NAME}.pth")
MODEL_SAVE_PATH_ONNX = os.path.join(SAVE_DIR, f"{MODEL_BASE_NAME}.onnx")
LOSS_CURVE_PATH = os.path.join(SAVE_DIR, "loss_curves_instantiated.png")
ACC_CURVE_PATH = os.path.join(SAVE_DIR, "mapped_accuracy_curve_instantiated.png")
CONFUSION_MATRIX_PATH = os.path.join(SAVE_DIR, "confusion_matrix_instantiated_mapped.png")

SAVE_BEST_MODEL_PTH = True
SAVE_FINAL_MODEL_ONNX = True
LOAD_SAVED_STATE = False

# --- Mappings (Unchanged) ---
LABEL_TO_IDX_MAP = {
    'Not Engaged': 0, 'Barely Engaged': 1, 'Engaged': 2, 'Highly Engaged': 3,
    'not engaged': 0, 'not-engaged': 0, 'Not-Engaged': 0,
    'barely engaged': 1, 'barely-engaged': 1, 'Barely-engaged': 1,
    'highly engaged': 3, 'highly-engaged': 3, 'Highly-Engaged': 3,
    'snp(subject not present)': 4, 'SNP(Subject Not Present)': 4, 'SNP': 4,
}
IDX_TO_SCORE_MAP = {4: 0.0, 0: 0.25, 1: 0.5, 2: 0.75, 3: 1.0} # SNP mapped to 0.0 score
IDX_TO_NAME_MAP = {0: 'Not Engaged', 1: 'Barely Engaged', 2: 'Engaged', 3: 'Highly Engaged', 4: 'SNP'}

# --- ONNX Export Settings (Unchanged) ---
ONNX_OPSET_VERSION = 11

# --- Print Configuration Function ---
def print_config():
    """Prints the configuration settings."""
    print("--- Configuration ---")
    print(f"Device: {DEVICE}")
    print(f"Config Path (for data): {CONFIG_PATH}")
    print(f"Input Dim: {INPUT_DIM}")
    print(f"Output Dim: {OUTPUT_DIM}")
    print("\nModel Architecture (Instantiated Layers):")
    # Try to print layer info more informatively
    for i, layer in enumerate(MODEL_LAYERS):
        # Get class name and basic info if possible
        info = f"Layer {i}: {layer.__class__.__name__}"
        if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
            info += f"(in={layer.in_features}, out={layer.out_features})"
        elif hasattr(layer, 'hidden_size'):
             info += f"(hidden_size={layer.hidden_size}, num_layers={getattr(layer, 'num_layers', '?')}, bidirectional={getattr(layer, 'bidirectional', '?')})"
        elif hasattr(layer, 'p'):
             info += f"(p={layer.p})"
        print(f"  {info}")

    print(f"\nLoad Saved State: {LOAD_SAVED_STATE}")
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


if __name__ == '__main__':
    print_config()
