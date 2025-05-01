# Model/model_config_gru_attn.py

import os
import torch
import torch.nn as nn
# --- Import the NEW GRU Attention model ---
from Model.models.gru_attention_model import GruAttentionModel

# ================================================
# === Configuration GRU Attention ===
# ================================================
CONFIG_PATH = "./Preprocess/Pipeline/Pipelines/configs/ENGAGENET_10fps_quality95_randdist.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model & Training Hyperparameters ---
# Start with similar params as the successful simplified LSTM run
LEARNING_RATE = 0.00005
BATCH_SIZE = 32
NUM_EPOCHS = 100 # Let early stopping (implicit via save best) decide
WEIGHT_DECAY = 1e-4

# --- Saving & Loading ---
MODEL_BASE_NAME = "v4_gru_attention" # New version name
SAVE_DIR = f"./saved_models/{MODEL_BASE_NAME}/"

# Construct full paths
MODEL_SAVE_PATH_PTH = os.path.join(SAVE_DIR, f"{MODEL_BASE_NAME}.pth")
MODEL_SAVE_PATH_ONNX = os.path.join(SAVE_DIR, f"{MODEL_BASE_NAME}.onnx")
LOSS_CURVE_PATH = os.path.join(SAVE_DIR, "loss_curves.png")
ACC_CURVE_PATH = os.path.join(SAVE_DIR, "mapped_accuracy_curve.png")
CONFUSION_MATRIX_PATH = os.path.join(SAVE_DIR, "confusion_matrix_regression_mapped.png")

SAVE_BEST_MODEL_PTH = True
SAVE_FINAL_MODEL_ONNX = True
LOAD_SAVED_STATE = True # Set to False to force training from scratch

# --- Mappings ---
# Use the adjusted score map that seemed to help slightly
LABEL_TO_IDX_MAP = {
    'Not Engaged': 0, 'Barely Engaged': 1, 'Engaged': 2, 'Highly Engaged': 3,
    'not engaged': 0, 'not-engaged': 0, 'Not-Engaged': 0,
    'barely engaged': 1, 'barely-engaged': 1, 'Barely-engaged': 1,
    'highly engaged': 3, 'highly-engaged': 3, 'Highly-Engaged': 3,
    'snp(subject not present)': 4, 'SNP(Subject Not Present)': 4, 'SNP': 4,
}
IDX_TO_SCORE_MAP = {
    4: 0.05, 0: 0.30, 1: 0.50, 2: 0.70, 3: 0.95
}
IDX_TO_NAME_MAP = {0: 'Not Engaged', 1: 'Barely Engaged', 2: 'Engaged', 3: 'Highly Engaged', 4: 'SNP'}

# --- ONNX Export Settings ---
ONNX_OPSET_VERSION = 11 # Usually fine, adjust if export errors occur


def get_model():
    """ Instantiates and returns the GRU Attention model. """
    return GruAttentionModel()


# --- Print Configuration Function ---
def print_config():
    """Prints the GRU Attention configuration settings."""
    print("--- Configuration V4 (GRU Attention) ---") # Updated title
    print(f"Device: {DEVICE}")
    print(f"Config Path: {CONFIG_PATH}")
    print(f"Model Base Name: {MODEL_BASE_NAME}")
    print(f"Load Saved State: {LOAD_SAVED_STATE}")
    print(f"Save Best PyTorch Model (.pth): {SAVE_BEST_MODEL_PTH}")
    if SAVE_BEST_MODEL_PTH or LOAD_SAVED_STATE:
        print(f"  PyTorch Model Path: {MODEL_SAVE_PATH_PTH}")
    print(f"Save Final ONNX Model (.onnx): {SAVE_FINAL_MODEL_ONNX}")
    if SAVE_FINAL_MODEL_ONNX:
        print(f"  ONNX Model Path: {MODEL_SAVE_PATH_ONNX}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Target Score Range: approx [0.05, 0.95]")
    print("-------------------")
    print("Label -> Index -> Score Mapping Initialized.")
    print("-------------------")

if __name__ == '__main__':
    print_config()