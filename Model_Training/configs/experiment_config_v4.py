# engagement_hf_trainer/configs/experiment_config.py
from typing import Optional

import torch
import torch.nn as nn

from Model_Training.configs.mesh_flipmap import mesh_annotations_derived_flip_map
from Model_Training.models.multitask_gru_attention_model_v4 import EngagementMultiTaskGRUAttentionModel
from Model_Training.pipelines.pipeline import OrchestrationPipeline
from Model_Training.pipelines.stages.DataAugmentationStage import DataAugmentationStage
from Model_Training.pipelines.stages.DistanceNormalizationStage import DistanceNormalizationStage
from Model_Training.pipelines.stages.label_processor_stage import LabelProcessorStage


EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_THRESHOLD = 0.001

TRAINING_HYPERPARAMS = {
    "num_train_epochs": 50,
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 124,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "logging_strategy": "steps",
    "logging_steps": 100,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 3, # Keep N best/recent checkpoints + final best model
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_mae", # Ensure 'eval_' prefix matches Trainer's logs
    "greater_is_better": False, # For MAE (lower is better)
    "fp16": True, # Enable mixed precision if CUDA is available
    "dataloader_num_workers": 4, # Adjust based on your CPU cores / system
    "dataloader_pin_memory": True,
    "report_to": "tensorboard", # Options: "tensorboard", "wandb", "none"
}

# --- Label Mappings (Crucial for data processing and evaluation) ---
LABEL_TO_IDX_MAP = {
    'Not Engaged': 0, 'Barely Engaged': 1, 'Engaged': 2, 'Highly Engaged': 3,
    'not engaged': 0, 'not-engaged': 0, 'Not-Engaged': 0,
    'barely engaged': 1, 'barely-engaged': 1, 'Barely-engaged': 1,
    'highly engaged': 3, 'highly-engaged': 3, 'Highly-Engaged': 3,
    'snp(subject not present)': 4, 'SNP(Subject Not Present)': 4, 'SNP': 4,
}

IDX_TO_SCORE_MAP = {
    # Index: Score
    4: 0.05,  # SNP
    0: 0.30,  # Not Engaged
    1: 0.50,  # Barely Engaged
    2: 0.70,  # Engaged
    3: 0.95   # Highly Engaged
}

IDX_TO_NAME_MAP = {
    0: 'Not Engaged',
    1: 'Barely Engaged',
    2: 'Engaged',
    3: 'Highly Engaged',
    4: 'SNP'
}

NUM_CLASSES_CLASSIFICATION = 5 # Number of unique classes (0, 1, 2, 3, 4)
ENGAGEMENT_KEY_IN_RAW_LABEL = 'engagement_string' # Key in the raw label dict from .pt files

# --- Model Configuration ---
MODEL_CLASS = EngagementMultiTaskGRUAttentionModel # The class of the model to use

MODEL_PARAMS = {
    # "input_dim" will be calculated in run_training.py based on dataset landmarks
    "hidden_dim": 256,
    "num_gru_layers": 2,
    "dropout_rate": 0.4,
    "bidirectional_gru": True,
    "regression_output_dim": 1,
    "num_classes": NUM_CLASSES_CLASSIFICATION,
    "regression_loss_weight": 1.0,
    "classification_loss_weight": 0.5,
}

NUM_COORDINATES = 3 # Usually 3 for (X, Y, Z)

# --- Loss Functions ---
# Instantiated here to be passed to the model
REGRESSION_LOSS_FN = nn.MSELoss()
CLASSIFICATION_LOSS_FN = nn.CrossEntropyLoss()

# --- Training Arguments related paths (can be overridden by TrainingArguments in run_script) ---
BASE_OUTPUT_DIR = "./training_runs_output/"
EXPERIMENT_NAME = "engagement_multitask_v4" # Used to create a subdirectory in BASE_OUTPUT_DIR

# --- Paths for Custom Callbacks (ONNX, Plots) ---
# Filenames; they will be joined with the Trainer's output_dir in the run_script
LOSS_CURVE_FILENAME = "loss_curves.png"
METRICS_CURVE_FILENAME = "metrics_curves.png" # For other metrics like MAE, R2, cls_accuracy
LR_CURVE_FILENAME = "learning_rate_curve.png"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"


MESH_FLIP_MAP = mesh_annotations_derived_flip_map


DATA_AUGMENTATION_PARAMS = {
    "add_noise_prob": 0.5,
    "noise_std": 0.01, # Adjusted from 0.02 in original config
    "random_scale_prob": 0.5,
    "scale_range": (0.9, 1.1), # Adjusted from (0.95, 1.05)
    "random_rotate_prob": 0.5,
    "max_rotation_angle_deg": 10.0,
    "random_flip_prob": 0.5,
    "landmark_flip_map": MESH_FLIP_MAP,
    "verbose": False
}

# Instantiate the LabelProcessorStage using the maps
label_processor_stage_instance = LabelProcessorStage(
    label_to_idx_map=LABEL_TO_IDX_MAP,
    idx_to_score_map=IDX_TO_SCORE_MAP,
    engagement_key=ENGAGEMENT_KEY_IN_RAW_LABEL,
    verbose=False # Set to True for debugging this stage
)

# Define default landmark indices for DistanceNormalizationStage
# IMPORTANT: User MUST verify these for their specific 478 landmark model.
# These are common for MediaPipe 468 landmarks.
DEFAULT_NOSE_TIP_IDX = 1
DEFAULT_LEFT_EYE_OUTER_IDX = 33 # Example, might be different for 478 landmarks
DEFAULT_RIGHT_EYE_OUTER_IDX = 263 # Example, might be different for 478 landmarks

distance_normalization_stage_instance = DistanceNormalizationStage(
    nose_tip_index=DEFAULT_NOSE_TIP_IDX,
    left_eye_outer_corner_index=DEFAULT_LEFT_EYE_OUTER_IDX,
    right_eye_outer_corner_index=DEFAULT_RIGHT_EYE_OUTER_IDX,
    verbose=False
)

data_augmentation_stage_instance = DataAugmentationStage(**DATA_AUGMENTATION_PARAMS)

# --- Define Train, Validation, and Test Pipelines ---
TRAIN_PIPELINE = OrchestrationPipeline(
    stages=[
        label_processor_stage_instance,
        distance_normalization_stage_instance,
        # data_augmentation_stage_instance,
        # Add other stages for training as needed
    ]
)

VALIDATION_PIPELINE = OrchestrationPipeline(
    stages=[
        label_processor_stage_instance,
        distance_normalization_stage_instance,
        # No augmentation for validation
    ]
)

TEST_PIPELINE = OrchestrationPipeline(
    stages=[
        label_processor_stage_instance,
        distance_normalization_stage_instance,
        # No augmentation for test
    ]
)

REPRESENTATIVE_SEQ_LEN_FOR_ONNX = 30 # Example, adjust based on your data
ONNX_MODEL_FILENAME = f"{EXPERIMENT_NAME}.onnx" # Just the filename
ONNX_OPSET_VERSION = 11
ONNX_EXPORT_PARAMS = {
    "onnx_model_filename": ONNX_MODEL_FILENAME,
    "opset_version": ONNX_OPSET_VERSION,
    "input_names": ["input_x"],
    "output_names": ["regression_scores", "classification_logits"],
    "dynamic_axes": {
        "input_x": {0: "batch_size", 1: "sequence_length"},
        "regression_scores": {0: "batch_size"},
        "classification_logits": {0: "batch_size"},
    },
    # base_input_shape (seq_len, num_landmarks, num_coords) will be fully defined in run_training.py
    "representative_seq_len": REPRESENTATIVE_SEQ_LEN_FOR_ONNX
}

PLOTTING_CALLBACK_PARAMS = {
    "loss_plot_filename": "loss_curves.png",
    "lr_plot_filename": "learning_rate_curve.png",
    "regression_metrics_plot_filename": "regression_metrics.png",
    "classification_metrics_plot_filename": "classification_metrics.png"
    # CONFUSION_MATRIX_FILENAME (old) = "confusion_matrix.png" # Not currently plotted by callback
}


LOAD_INITIAL_WEIGHTS_PATH: Optional[str] = "./training_runs_output/engagement_multitask_v4/final_exported_models/model.safetensors"
SAVE_FINAL_PYTORCH_MODEL: bool = True
PERFORM_ONNX_EXPORT: bool = True

TRAINER_ARTIFACTS_SUBDIR_NAME = "trainer_artifacts"
FINAL_MODELS_SUBDIR_NAME = "final_exported_models"
PLOTS_PARENT_SUBDIR_NAME = "training_plots"

if __name__ == '__main__':
    print("--- Experiment Configuration Loaded ---")
    print(f"Model Class: {MODEL_CLASS.__name__}")
    print(f"Number of classes for classification: {NUM_CLASSES_CLASSIFICATION}")
    print(f"Regression Loss Weight: {MODEL_PARAMS['regression_loss_weight']}")
    print(f"Classification Loss Weight: {MODEL_PARAMS['classification_loss_weight']}")
    print(f"IDX_TO_SCORE_MAP for SNP (idx 4): {IDX_TO_SCORE_MAP.get(4)}")
    print(f"Train pipeline has {len(TRAIN_PIPELINE.stages)} stages.")
    print(f"First stage in train pipeline: {TRAIN_PIPELINE.stages[0].__class__.__name__}")
