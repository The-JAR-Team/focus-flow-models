# Model/model_config_gru_attn.py

import os
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim

from Model.mesh_flipmap import mesh_annotations_derived_flip_map
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
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-4
CRITERION = nn.MSELoss()
OPTIMIZER_CLASS = torch.optim.AdamW
MODEL = GruAttentionModel()

LR_SCHEDULER_TYPE = "StepLR"  # Or "ReduceLROnPlateau" or None

# Parameters for StepLR
STEP_LR_STEP_SIZE = 30
STEP_LR_GAMMA = 0.1

# Parameters for ReduceLROnPlateau
REDUCE_LR_PATIENCE = 10
REDUCE_LR_FACTOR = 0.1
REDUCE_LR_MIN_LR = 1e-6
REDUCE_LR_VERBOSE = True

# --- Saving & Loading ---
MODEL_BASE_NAME = "v3_gru_attention" # New version name
SAVE_DIR = f"./saved_models/{MODEL_BASE_NAME}/"

# Construct full paths
MODEL_SAVE_PATH_PTH = os.path.join(SAVE_DIR, f"{MODEL_BASE_NAME}.pth")
MODEL_SAVE_PATH_ONNX = os.path.join(SAVE_DIR, f"{MODEL_BASE_NAME}.onnx")
LOSS_CURVE_PATH = os.path.join(SAVE_DIR, "loss_curves.png")
ACC_CURVE_PATH = os.path.join(SAVE_DIR, "mapped_accuracy_curve.png")
CONFUSION_MATRIX_PATH = os.path.join(SAVE_DIR, "confusion_matrix_regression_mapped.png")
LR_CURVE_PATH = os.path.join(SAVE_DIR, "learning_rate_curve.png")


SAVE_BEST_MODEL_PTH = True
SAVE_FINAL_MODEL_ONNX = True
LOAD_SAVED_STATE = True # Set to False to force training from scratch

TRAIN_DATALOADER_PIPELINE = OrchestrationPipeline(
            stages=[
                DistanceNormalizationStage(verbose=False),
                DataAugmentationStage(
                    add_noise_prob=0.5, noise_std=0.02,
                    random_scale_prob=0.5, scale_range=(0.95, 1.05),
                    random_rotate_prob=0.5, max_rotation_angle_deg=10,
                    random_flip_prob=0.5, landmark_flip_map=mesh_annotations_derived_flip_map,
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


def get_lr_scheduler(optimizer_instance: optim.Optimizer):
    """
    Initializes and returns the learning rate scheduler based on config.
    Args:
        optimizer_instance: The instantiated optimizer.
    Returns:
        A learning rate scheduler instance or None.
    """
    if LR_SCHEDULER_TYPE == "StepLR":
        print(f"Initializing StepLR scheduler with step_size={STEP_LR_STEP_SIZE}, gamma={STEP_LR_GAMMA}")
        return lr_scheduler.StepLR(optimizer_instance,
                                   step_size=STEP_LR_STEP_SIZE,
                                   gamma=STEP_LR_GAMMA)
    elif LR_SCHEDULER_TYPE == "ReduceLROnPlateau":
        print(f"Initializing ReduceLROnPlateau scheduler with factor={REDUCE_LR_FACTOR}, patience={REDUCE_LR_PATIENCE}")
        return lr_scheduler.ReduceLROnPlateau(optimizer_instance,
                                              mode='min',
                                              factor=REDUCE_LR_FACTOR,
                                              patience=REDUCE_LR_PATIENCE,
                                              min_lr=REDUCE_LR_MIN_LR,
                                              verbose=REDUCE_LR_VERBOSE)
    elif LR_SCHEDULER_TYPE is None:
        print("No LR scheduler will be used.")
        return None
    else:
        print(f"Warning: Unknown LR_SCHEDULER_TYPE: {LR_SCHEDULER_TYPE}. No scheduler will be used.")
        return None


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