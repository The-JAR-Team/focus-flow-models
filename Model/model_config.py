# Model/model_config_gru_attn.py

import os
import torch
import torch.nn as nn
# --- Import the NEW GRU Attention model ---
from Model.models.gru_attention_model import GruAttentionModel
from Preprocess.Pipeline.OrchestrationPipeline import OrchestrationPipeline
from Preprocess.Pipeline.Stages.dataloader_stages.DataAugmentationStage import DataAugmentationStage
from Preprocess.Pipeline.Stages.dataloader_stages.DistanceNormalizationStage import DistanceNormalizationStage

# ================================================
# === Configuration GRU Attention ===
# ================================================
CONFIG_PATH = "./Preprocess/Pipeline/Pipelines/configs/ENGAGENET_10fps_quality95_randdist.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model & Training Hyperparameters ---
# Start with similar params as the successful simplified LSTM run
INPUT_DIM = 478 * 3 # Example: (num_landmarks * coordinates) - Adjust if needed
LEARNING_RATE = 0.00005
BATCH_SIZE = 32
NUM_EPOCHS = 10
WEIGHT_DECAY = 1e-4
CRITERION = nn.MSELoss()
OPTIMIZER = torch.optim.AdamW
MODEL = GruAttentionModel()

# --- Saving & Loading ---
MODEL_BASE_NAME = "v3_gru_attention" # New version name
SAVE_DIR = f"./saved_models/{MODEL_BASE_NAME}/"

# Construct full paths
MODEL_SAVE_PATH_PTH = os.path.join(SAVE_DIR, f"{MODEL_BASE_NAME}.pth")
MODEL_SAVE_PATH_ONNX = os.path.join(SAVE_DIR, f"{MODEL_BASE_NAME}.onnx")
LOSS_CURVE_PATH = os.path.join(SAVE_DIR, "loss_curves.png")
ACC_CURVE_PATH = os.path.join(SAVE_DIR, "mapped_accuracy_curve.png")
CONFUSION_MATRIX_PATH = os.path.join(SAVE_DIR, "confusion_matrix_regression_mapped.png")

SAVE_BEST_MODEL_PTH = False
SAVE_FINAL_MODEL_ONNX = False
LOAD_SAVED_STATE = True # Set to False to force training from scratch

TRAIN_DATALOADER_PIPELINE = OrchestrationPipeline(
            stages=[
                DistanceNormalizationStage(verbose=False),
                DataAugmentationStage(
                    add_noise_prob=0.5, noise_std=0.02,
                    random_scale_prob=0.5, scale_range=(0.95, 1.05),
                    random_rotate_prob=0.5, max_rotation_angle_deg=10,
                    random_flip_prob=0.5,
                    verbose=False
                ),
            ],
        )
VALIDATION_DATALOADER_PIPELINE = OrchestrationPipeline(
            stages=[
                DistanceNormalizationStage(verbose=False),
            ],
        )
TEST_DATALOADER_PIPELINE = OrchestrationPipeline(
            stages=[
                DistanceNormalizationStage(verbose=False),
            ],
        )

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