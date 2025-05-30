# engagement_hf_trainer/configs/experiment_config_v4_v2_finetune.py
from typing import Optional, Dict, Union

import torch
import torch.nn as nn

from Model_Training.configs.mesh_flipmap import mesh_annotations_derived_flip_map
from Model_Training.models.multitask_gru_attention_model_v4_2 import EngagementMultiTaskGRUAttentionModel # Ensure this model has the forward() fix
from Model_Training.pipelines.pipeline import OrchestrationPipeline
from Model_Training.pipelines.stages.DataAugmentationStage import DataAugmentationStage
from Model_Training.pipelines.stages.DistanceNormalizationStage import DistanceNormalizationStage
from Model_Training.pipelines.stages.label_processor_stage import LabelProcessorStage

# --- Path to Load Best Weights From ---
# !!! IMPORTANT: FILL THIS IN with the path to your best v4_v2 model's .safetensors or .bin file !!!
# Example: "./training_runs_output/engagement_multitask_v4_v2/final_exported_models/model.safetensors"
LOAD_INITIAL_WEIGHTS_PATH: Optional[str] = "./training_runs_output/engagement_multitask_v4_2/final_exported_models/model_40_20_25.safetensors"


# --- Early Stopping Parameters ---
EARLY_STOPPING_PATIENCE = 25 # Patience for fine-tuning
EARLY_STOPPING_THRESHOLD = 0.0005 # Slightly finer threshold for fine-tuning

# --- Training Hyperparameters ---
TRAINING_HYPERPARAMS = {
    "num_train_epochs": 25,  # Fine-tune for 25 epochs
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 124,
    "learning_rate": 1.5e-5,  # Reduced learning rate for fine-tuning
    "warmup_ratio": 0.05, # Reduced warmup for fine-tuning, or 0 if loading optimizer state is off
    "weight_decay": 0.01,
    "logging_strategy": "steps",
    "logging_steps": 50,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_mae",
    "greater_is_better": False,
    "fp16": True,
    "dataloader_num_workers": 0,
    "dataloader_pin_memory": True,
    "report_to": "tensorboard",
}

# --- Label Mappings (Same as previous) ---
LABEL_TO_IDX_MAP: Dict[str, int] = {
    'Not Engaged': 0, 'Barely Engaged': 1, 'Engaged': 2, 'Highly Engaged': 3,
    'not engaged': 0, 'not-engaged': 0, 'Not-Engaged': 0,
    'barely engaged': 1, 'barely-engaged': 1, 'Barely-engaged': 1,
    'highly engaged': 3, 'highly-engaged': 3, 'Highly-Engaged': 3,
    'snp(subject not present)': 4, 'SNP(Subject Not Present)': 4, 'SNP': 4,
}
IDX_TO_SCORE_MAP: Dict[int, float] = {
    4: 0.05, 0: 0.30, 1: 0.50, 2: 0.70, 3: 0.95
}
IDX_TO_NAME_MAP: Dict[int, str] = {
    0: 'Not Engaged', 1: 'Barely Engaged', 2: 'Engaged', 3: 'Highly Engaged', 4: 'SNP'
}
NUM_CLASSES_CLASSIFICATION = 5
ENGAGEMENT_KEY_IN_RAW_LABEL = 'engagement_string'

# --- Model Configuration (Same as previous) ---
MODEL_CLASS = EngagementMultiTaskGRUAttentionModel
MODEL_PARAMS = {
    "hidden_dim": 256,
    "num_gru_layers": 2,
    "dropout_rate": 0.4, # Could slightly reduce for fine-tuning (e.g., 0.3) if overfitting was an issue
    "bidirectional_gru": True,
    "regression_output_dim": 1,
    "num_classes": NUM_CLASSES_CLASSIFICATION,
    "regression_loss_weight": 1.0,
    "classification_loss_weight": 0.5,
}
NUM_COORDINATES = 3

# --- Loss Functions (Same as previous) ---
REGRESSION_LOSS_FN = nn.MSELoss()
CLASSIFICATION_LOSS_FN = nn.CrossEntropyLoss()

# --- Paths and Naming ---
BASE_OUTPUT_DIR = "./training_runs_output/"
# New experiment name for this fine-tuning phase
EXPERIMENT_NAME = "engagement_multitask_v4_2"

# --- Data Augmentation Parameters (Kept same mild settings) ---
MESH_FLIP_MAP = mesh_annotations_derived_flip_map
DATA_AUGMENTATION_PARAMS = {
    "add_noise_prob": 0.1, "noise_std": 0.0005,
    "random_scale_prob": 0.1, "scale_range": (0.97, 1.03),
    "random_rotate_prob": 0.15, "max_rotation_angle_deg": 10.0,
    "random_flip_prob": 0.1, "landmark_flip_map": MESH_FLIP_MAP,
    "verbose": False
}

# --- Pipeline Stages Instantiation (Same as previous) ---
label_processor_stage_instance = LabelProcessorStage(
    label_to_idx_map=LABEL_TO_IDX_MAP, idx_to_score_map=IDX_TO_SCORE_MAP,
    engagement_key=ENGAGEMENT_KEY_IN_RAW_LABEL, verbose=False
)
DEFAULT_NOSE_TIP_IDX, DEFAULT_LEFT_EYE_OUTER_IDX, DEFAULT_RIGHT_EYE_OUTER_IDX = 1, 33, 263
distance_normalization_stage_instance = DistanceNormalizationStage(
    nose_tip_index=DEFAULT_NOSE_TIP_IDX, left_eye_outer_corner_index=DEFAULT_LEFT_EYE_OUTER_IDX,
    right_eye_outer_corner_index=DEFAULT_RIGHT_EYE_OUTER_IDX, verbose=False
)
data_augmentation_stage_instance = DataAugmentationStage(**DATA_AUGMENTATION_PARAMS)

TRAIN_PIPELINE = OrchestrationPipeline(stages=[
    label_processor_stage_instance, distance_normalization_stage_instance, data_augmentation_stage_instance
])
VALIDATION_PIPELINE = OrchestrationPipeline(stages=[
    label_processor_stage_instance, distance_normalization_stage_instance
])
TEST_PIPELINE = OrchestrationPipeline(stages=[
    label_processor_stage_instance, distance_normalization_stage_instance
])

# --- ONNX Export Configuration (Same as previous) ---
REPRESENTATIVE_SEQ_LEN_FOR_ONNX = 30
ONNX_MODEL_FILENAME = f"{EXPERIMENT_NAME}.onnx"
ONNX_OPSET_VERSION = 11
ONNX_EXPORT_PARAMS = {
    "onnx_model_filename": ONNX_MODEL_FILENAME, "opset_version": ONNX_OPSET_VERSION,
    "input_names": ["input_x"],
    "output_names": ["regression_scores", "classification_logits", "attention_weights"],
    "dynamic_axes": {
        "input_x": {0: "batch_size", 1: "sequence_length"},
        "regression_scores": {0: "batch_size"}, "classification_logits": {0: "batch_size"},
        "attention_weights": {0: "batch_size", 1: "sequence_length"},
    },
    "representative_seq_len": REPRESENTATIVE_SEQ_LEN_FOR_ONNX
}

# --- Plotting and Callbacks (Same as previous) ---
CONFUSION_MATRIX_EVAL_FILENAME = "cm_eval.png"
CONFUSION_MATRIX_TEST_FILENAME = "cm_test.png"
PLOTTING_CALLBACK_PARAMS = {
    "loss_plot_filename": "training_validation_loss.png", "lr_plot_filename": "learning_rate.png",
    "regression_metrics_plot_filename": "regression_metrics.png",
    "classification_metrics_plot_filename": "classification_metrics.png",
    "confusion_matrix_eval_filename": CONFUSION_MATRIX_EVAL_FILENAME,
    "confusion_matrix_test_filename": CONFUSION_MATRIX_TEST_FILENAME,
}

# --- Model Loading/Saving ---
# LOAD_INITIAL_WEIGHTS_PATH is set at the top of the file.
# RESUME_FROM_CHECKPOINT should be False or None for this fine-tuning strategy
# to ensure we are not restoring optimizer state, allowing the new LR to take full effect.
RESUME_FROM_CHECKPOINT: Union[str, bool, None] = False


SAVE_FINAL_PYTORCH_MODEL: bool = True
PERFORM_ONNX_EXPORT: bool = True

# --- Directory Names for Artifacts ---
TRAINER_ARTIFACTS_SUBDIR_NAME = "trainer_artifacts"
FINAL_MODELS_SUBDIR_NAME = "final_exported_models"
PLOTS_PARENT_SUBDIR_NAME = "training_plots"

if __name__ == '__main__':
    print(f"--- Experiment Configuration Loaded: {EXPERIMENT_NAME} ---")
    if LOAD_INITIAL_WEIGHTS_PATH and "YOUR_BEST_MODEL_FILE" not in LOAD_INITIAL_WEIGHTS_PATH:
        print(f"Loading initial weights for fine-tuning from: {LOAD_INITIAL_WEIGHTS_PATH}")
    else:
        print(f"Warning: LOAD_INITIAL_WEIGHTS_FROM_V4_V2_BEST placeholder not filled. Training from scratch or random weights if path is invalid.")
    print(f"RESUME_FROM_CHECKPOINT is set to: {RESUME_FROM_CHECKPOINT}")
    print(f"Fine-tuning for {TRAINING_HYPERPARAMS['num_train_epochs']} epochs with LR: {TRAINING_HYPERPARAMS['learning_rate']}.")

