# engagement_hf_trainer/configs/experiment_config_v4_v2_resumed.py
from typing import Optional, Dict, Union # Added Union

import torch
import torch.nn as nn

from Model_Training.configs.mesh_flipmap import mesh_annotations_derived_flip_map
# Ensure this model file (multitask_gru_attention_model_v4.py) has the forward() method fix
# for outputting attention_weights.
from Model_Training.models.multitask_gru_attention_model_v4 import EngagementMultiTaskGRUAttentionModel
from Model_Training.pipelines.pipeline import OrchestrationPipeline
from Model_Training.pipelines.stages.DataAugmentationStage import DataAugmentationStage
from Model_Training.pipelines.stages.DistanceNormalizationStage import DistanceNormalizationStage
from Model_Training.pipelines.stages.label_processor_stage import LabelProcessorStage

# --- Path to Resume From ---
# !!! IMPORTANT: FILL THIS IN with the path to your v4_v2 checkpoint !!!
# Example: "./training_runs_output/engagement_multitask_v4_v2/trainer_artifacts/checkpoint-3400"
# This path will be used to resume the training session.
RESUME_FROM_CHECKPOINT_PATH_PREVIOUS_RUN: Optional[str] = None
# --- Early Stopping Parameters ---
EARLY_STOPPING_PATIENCE = 7 # Adjusted for continuing training
EARLY_STOPPING_THRESHOLD = 0.001

# --- Training Hyperparameters ---
TRAINING_HYPERPARAMS = {
    "num_train_epochs": 30,  # Continue for 30 more epochs
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 124,
    "learning_rate": 5e-5,  # Kept same as v4_v2, scheduler will resume
    "warmup_ratio": 0.1, # Scheduler will likely have passed warmup if resuming
    "weight_decay": 0.01,
    "logging_strategy": "steps",
    "logging_steps": 50,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 4, # Increased slightly, total checkpoints from v4_v2 + this run
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_mae",
    "greater_is_better": False,
    "fp16": True,
    "dataloader_num_workers": 0, # Adjust if your system supports more
    "dataloader_pin_memory": True,
    "report_to": "tensorboard",
}

# --- Label Mappings ---
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

# --- Model Configuration ---
MODEL_CLASS = EngagementMultiTaskGRUAttentionModel
MODEL_PARAMS = {
    "hidden_dim": 256,
    "num_gru_layers": 2,
    "dropout_rate": 0.4,
    "bidirectional_gru": True,
    "regression_output_dim": 1,
    "num_classes": NUM_CLASSES_CLASSIFICATION,
    "regression_loss_weight": 1.0,
    "classification_loss_weight": 0.5,
}
NUM_COORDINATES = 3

# --- Loss Functions ---
REGRESSION_LOSS_FN = nn.MSELoss()
CLASSIFICATION_LOSS_FN = nn.CrossEntropyLoss()

# --- Paths and Naming ---
BASE_OUTPUT_DIR = "./training_runs_output/"
EXPERIMENT_NAME = "engagement_multitask_v4_v2" # Reverted to v4_v2 for continuation

# --- Data Augmentation Parameters ---
MESH_FLIP_MAP = mesh_annotations_derived_flip_map
DATA_AUGMENTATION_PARAMS = {
    "add_noise_prob": 0.1, "noise_std": 0.0005,
    "random_scale_prob": 0.1, "scale_range": (0.97, 1.03),
    "random_rotate_prob": 0.15, "max_rotation_angle_deg": 10.0,
    "random_flip_prob": 0.1, "landmark_flip_map": MESH_FLIP_MAP,
    "verbose": False
}

# --- Pipeline Stages Instantiation ---
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

# --- ONNX Export Configuration ---
REPRESENTATIVE_SEQ_LEN_FOR_ONNX = 30
ONNX_MODEL_FILENAME = f"{EXPERIMENT_NAME}.onnx" # Uses the updated EXPERIMENT_NAME
ONNX_OPSET_VERSION = 11
ONNX_EXPORT_PARAMS = {
    "onnx_model_filename": ONNX_MODEL_FILENAME, "opset_version": ONNX_OPSET_VERSION,
    "input_names": ["input_x"],
    "output_names": ["regression_scores", "classification_logits", "attention_weights"], # Must match model's forward() output dict keys
    "dynamic_axes": {
        "input_x": {0: "batch_size", 1: "sequence_length"},
        "regression_scores": {0: "batch_size"}, "classification_logits": {0: "batch_size"},
        "attention_weights": {0: "batch_size", 1: "sequence_length"}, # Ensure model outputs this
    },
    "representative_seq_len": REPRESENTATIVE_SEQ_LEN_FOR_ONNX
}

# --- Plotting and Callbacks ---
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
# LOAD_INITIAL_WEIGHTS_PATH should be None if RESUME_FROM_CHECKPOINT is used effectively.
# The Hugging Face Trainer handles loading weights, optimizer, and scheduler from the checkpoint.
LOAD_INITIAL_WEIGHTS_PATH: Optional[str] = './training_runs_output/engagement_multitask_v4_v2/final_exported_models/model_40.safetensors'

# This variable should be used by your run_training.py script to pass to trainer.train(resume_from_checkpoint=...)
# It will be the path string if filled, or False if the placeholder isn't filled (to prevent errors).
# If you want to auto-detect the latest checkpoint in the experiment's output_dir, your run_training.py
# would need to handle setting this to `True` (boolean) or finding the latest checkpoint path itself.
RESUME_FROM_CHECKPOINT: Union[str, bool, None] = RESUME_FROM_CHECKPOINT_PATH_PREVIOUS_RUN \
    if RESUME_FROM_CHECKPOINT_PATH_PREVIOUS_RUN and "YOUR_LAST_CHECKPOINT_STEP" not in RESUME_FROM_CHECKPOINT_PATH_PREVIOUS_RUN \
    else False


SAVE_FINAL_PYTORCH_MODEL: bool = True
PERFORM_ONNX_EXPORT: bool = True # Assumes model's forward() method is fixed for attention_weights

# --- Directory Names for Artifacts ---
TRAINER_ARTIFACTS_SUBDIR_NAME = "trainer_artifacts" # Checkpoints will go here
FINAL_MODELS_SUBDIR_NAME = "final_exported_models"
PLOTS_PARENT_SUBDIR_NAME = "training_plots"

if __name__ == '__main__':
    print(f"--- Experiment Configuration Loaded: {EXPERIMENT_NAME} (Resumed/Extended) ---")
    if RESUME_FROM_CHECKPOINT and isinstance(RESUME_FROM_CHECKPOINT, str):
        print(f"Attempting to resume training from checkpoint: {RESUME_FROM_CHECKPOINT}")
    elif RESUME_FROM_CHECKPOINT is True:
        print(f"Attempting to resume from the latest checkpoint in the output directory for {EXPERIMENT_NAME}.")
    else:
        print("Not resuming from a specific checkpoint (RESUME_FROM_CHECKPOINT is False or None). Training might start fresh or load from LOAD_INITIAL_WEIGHTS_PATH if set.")
    print(f"Training for {TRAINING_HYPERPARAMS['num_train_epochs']} additional epochs.")
    print(f"ONNX Filename: {ONNX_MODEL_FILENAME}")
