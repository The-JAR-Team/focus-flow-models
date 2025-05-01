import os
import torch
import torch.nn as nn
from Model.models.gru_model import GruModel

# ================================================
# === Configuration ===
# ================================================
CONFIG_PATH = "./Preprocess/Pipeline/Pipelines/configs/ENGAGENET_10fps_quality95_randdist.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model & Training Hyperparameters ---
LEARNING_RATE = 0.00001
BATCH_SIZE = 32
NUM_EPOCHS = 1

# --- Saving & Loading ---
# Base name for saved files (paths will be constructed)
MODEL_BASE_NAME = "engagement_model_regression_0_2"
SAVE_DIR = "." # Directory to save models and plots

# Construct full paths
MODEL_SAVE_PATH_PTH = os.path.join(SAVE_DIR, f"{MODEL_BASE_NAME}.pth")
MODEL_SAVE_PATH_ONNX = os.path.join(SAVE_DIR, f"{MODEL_BASE_NAME}.onnx")
LOSS_CURVE_PATH = os.path.join(SAVE_DIR, "loss_curves.png")
ACC_CURVE_PATH = os.path.join(SAVE_DIR, "mapped_accuracy_curve.png")
CONFUSION_MATRIX_PATH = os.path.join(SAVE_DIR, "confusion_matrix_regression_mapped.png")


SAVE_BEST_MODEL_PTH = False # Save best model state dict during training?
SAVE_FINAL_MODEL_ONNX = False # Save the final best model as ONNX after training?
LOAD_SAVED_STATE = True # Attempt to load MODEL_SAVE_PATH_PTH before training?

# --- Mappings ---
LABEL_TO_IDX_MAP = {
    'Not Engaged': 0, 'Barely Engaged': 1, 'Engaged': 2, 'Highly Engaged': 3,
    'not engaged': 0, 'not-engaged': 0, 'Not-Engaged': 0,
    'barely engaged': 1, 'barely-engaged': 1, 'Barely-engaged': 1,
    'highly engaged': 3, 'highly-engaged': 3, 'Highly-Engaged': 3,
    'snp(subject not present)': 4, 'SNP(Subject Not Present)': 4, 'SNP': 4,
}
IDX_TO_SCORE_MAP = {4: 0.0, 0: 0.25, 1: 0.5, 2: 0.75, 3: 1.0} # SNP mapped to 0.0 score
IDX_TO_NAME_MAP = {0: 'Not Engaged', 1: 'Barely Engaged', 2: 'Engaged', 3: 'Highly Engaged', 4: 'SNP'}

# --- ONNX Export Settings ---
ONNX_OPSET_VERSION = 11


def get_model():
    """
    Returns the model class for engagement regression.

    Returns:
        GruModel: The GRU-based model for engagement regression.
    """
    return GruModel()


# --- Print Configuration Function ---
def print_config():
    """Prints the configuration settings."""
    print("--- Configuration ---")
    print(f"Device: {DEVICE}")
    print(f"Config Path: {CONFIG_PATH}")
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
    print("Label -> Index -> Score Mapping Initialized.")
    print("-------------------")


if __name__ == '__main__':
    # Example of how to use print_config if running this file directly
    print_config()
