# engagement_hf_trainer/configs/experiment_config_v4_5.py
from typing import Optional, Dict, Union

import torch
import torch.nn as nn

from Model_Training.configs.mesh_flipmap import mesh_annotations_derived_flip_map
# Assuming multitask_gru_attention_model_v4_2 is the intended model file
from Model_Training.models.multitask_gru_attention_model_v4_2 import EngagementMultiTaskGRUAttentionModel
from Model_Training.pipelines.pipeline import OrchestrationPipeline
from Model_Training.pipelines.stages.DataAugmentationStage import DataAugmentationStage
from Model_Training.pipelines.stages.DistanceNormalizationStage import DistanceNormalizationStage
from Model_Training.pipelines.stages.label_processor_stage import LabelProcessorStage
from Model_Training.pipelines.stages.SNPAugmentationStage import SNPAugmentationStage


# --- Path to Load Initial Weights From (if any) ---
# Set to None for a fresh training run for v4_5.
LOAD_INITIAL_WEIGHTS_PATH: Optional[str] = None

# --- Early Stopping Parameters ---
EARLY_STOPPING_PATIENCE = 30 # Kept from v4_4
EARLY_STOPPING_THRESHOLD = 0.0005 # Kept from v4_4

# --- Training Hyperparameters (Same as v4_4) ---
TRAINING_HYPERPARAMS = {
    "num_train_epochs": 40,
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 128,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.1,
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

# --- Label Mappings (Consistent) ---
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

# --- Model Configuration (Same class) ---
MODEL_CLASS = EngagementMultiTaskGRUAttentionModel
MODEL_PARAMS = {
    "hidden_dim": 256,
    "num_gru_layers": 2,
    "dropout_rate": 0.4, # Consider if this needs adjustment with heavy aug
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
EXPERIMENT_NAME = "multitask_v4_5"  # New experiment name for v4_5

# --- Data Augmentation Parameters (Aggressively Increased Probabilities and Magnitudes) ---
MESH_FLIP_MAP = mesh_annotations_derived_flip_map
DATA_AUGMENTATION_PARAMS = {
    # Standard geometric augmentations - More Aggressive
    "add_noise_prob": 0.8, "noise_std": 0.008,
    "random_scale_prob": 0.8, "scale_range": (0.90, 1.10),
    "random_rotate_prob": 0.8, "max_rotation_angle_deg": 20.0,
    "random_flip_prob": 0.7, "landmark_flip_map": MESH_FLIP_MAP,

    # Temporal Displacement Jitter - More Aggressive
    "temporal_jitter_prob": 0.8,
    "jitter_burst_length_range": (5, 25),
    "jitter_magnitude_std": 0.15,
    "max_jitter_bursts_per_sequence": 4,

    "verbose": False
}

# --- SNP Augmentation Parameters (Aggressively Increased Probability) ---
SNP_CLASS_IDX = LABEL_TO_IDX_MAP['SNP']
SNP_REG_SCORE = IDX_TO_SCORE_MAP[SNP_CLASS_IDX]

SNP_AUGMENTATION_PARAMS = {
    "snp_conversion_prob": 0.05,
    "min_snp_frame_percentage": 0.4, # Increased to 40%
    "snp_class_idx": SNP_CLASS_IDX,
    "snp_reg_score": SNP_REG_SCORE,
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
snp_augmentation_stage_instance = SNPAugmentationStage(**SNP_AUGMENTATION_PARAMS)

# --- Pipeline Definitions (Same structure as v4_4) ---
TRAIN_PIPELINE = OrchestrationPipeline(stages=[
    label_processor_stage_instance,
    distance_normalization_stage_instance,
    data_augmentation_stage_instance,
    snp_augmentation_stage_instance
])

VALIDATION_PIPELINE = OrchestrationPipeline(stages=[
    label_processor_stage_instance,
    distance_normalization_stage_instance,
    snp_augmentation_stage_instance
])

TEST_PIPELINE = OrchestrationPipeline(stages=[
    label_processor_stage_instance,
    distance_normalization_stage_instance
])

# --- ONNX Export Configuration ---
REPRESENTATIVE_SEQ_LEN_FOR_ONNX = 30
ONNX_MODEL_FILENAME = f"{EXPERIMENT_NAME}.onnx"
ONNX_OPSET_VERSION = 11
ONNX_EXPORT_PARAMS = {
    "onnx_model_filename": ONNX_MODEL_FILENAME,
    "opset_version": ONNX_OPSET_VERSION,
    "input_names": ["input_x"],
    "output_names": ["regression_scores", "classification_logits", "attention_weights"],
    "dynamic_axes": {
        "input_x": {0: "batch_size", 1: "sequence_length"},
        "regression_scores": {0: "batch_size"},
        "classification_logits": {0: "batch_size"},
        "attention_weights": {0: "batch_size", 1: "sequence_length"},
    },
    "representative_seq_len": REPRESENTATIVE_SEQ_LEN_FOR_ONNX
}

# --- Plotting and Callbacks ---
CONFUSION_MATRIX_EVAL_FILENAME = "cm_eval.png"
CONFUSION_MATRIX_TEST_FILENAME = "cm_test.png"
PLOTTING_CALLBACK_PARAMS = {
    "loss_plot_filename": "training_validation_loss.png",
    "lr_plot_filename": "learning_rate.png",
    "regression_metrics_plot_filename": "regression_metrics.png",
    "classification_metrics_plot_filename": "classification_metrics.png",
    "confusion_matrix_eval_filename": CONFUSION_MATRIX_EVAL_FILENAME,
    "confusion_matrix_test_filename": CONFUSION_MATRIX_TEST_FILENAME,
}

# --- Model Loading/Saving ---
RESUME_FROM_CHECKPOINT: Union[str, bool, None] = False
SAVE_FINAL_PYTORCH_MODEL: bool = True
PERFORM_ONNX_EXPORT: bool = True

# --- Directory Names for Artifacts ---
TRAINER_ARTIFACTS_SUBDIR_NAME = "trainer_artifacts"
FINAL_MODELS_SUBDIR_NAME = "final_exported_models"
PLOTS_PARENT_SUBDIR_NAME = "training_plots"

if __name__ == '__main__':
    print(f"--- Experiment Configuration Loaded: {EXPERIMENT_NAME} ---")
    if LOAD_INITIAL_WEIGHTS_PATH:
        print(f"Attempting to load initial weights from: {LOAD_INITIAL_WEIGHTS_PATH}")
    else:
        print("Starting training from scratch (no initial weights path specified).")
    print(
        f"Training for {TRAINING_HYPERPARAMS['num_train_epochs']} epochs with LR: {TRAINING_HYPERPARAMS['learning_rate']}.")
    print(f"RESUME_FROM_CHECKPOINT is set to: {RESUME_FROM_CHECKPOINT}")
    print(f"Data Augmentation includes AGGRESSIVE Temporal Jitter and Geometric Augmentations:")
    print(f"  Noise Prob: {DATA_AUGMENTATION_PARAMS['add_noise_prob']*100}%, Noise Std: {DATA_AUGMENTATION_PARAMS['noise_std']}")
    print(f"  Scale Prob: {DATA_AUGMENTATION_PARAMS['random_scale_prob']*100}%, Scale Range: {DATA_AUGMENTATION_PARAMS['scale_range']}")
    print(f"  Rotate Prob: {DATA_AUGMENTATION_PARAMS['random_rotate_prob']*100}%, Max Angle: {DATA_AUGMENTATION_PARAMS['max_rotation_angle_deg']}")
    print(f"  Flip Prob: {DATA_AUGMENTATION_PARAMS['random_flip_prob']*100}%")
    print(f"  Temporal Jitter Probability (per burst): {DATA_AUGMENTATION_PARAMS['temporal_jitter_prob'] * 100}%")
    print(f"  Jitter Burst Length (frames): {DATA_AUGMENTATION_PARAMS['jitter_burst_length_range']}")
    print(f"  Jitter Magnitude STD: {DATA_AUGMENTATION_PARAMS['jitter_magnitude_std']}")
    print(f"  Max Jitter Bursts per Sequence: {DATA_AUGMENTATION_PARAMS['max_jitter_bursts_per_sequence']}")
    print(f"SNP Augmentation will be applied to TRAIN and VALIDATION sets with AGGRESSIVE probability:")
    print(f"  SNP Conversion Probability: {SNP_AUGMENTATION_PARAMS['snp_conversion_prob'] * 100}%")
    print(f"  Min SNP Frame Percentage: {SNP_AUGMENTATION_PARAMS['min_snp_frame_percentage']*100}%")
