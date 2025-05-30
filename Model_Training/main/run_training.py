# Model_Training/main/run_training.py
import os
import sys
import torch
import numpy as np
import functools
from datetime import datetime
from typing import Optional, Dict, Any, Union, Mapping, Any as AnyType  # Renamed Any to AnyType to avoid conflict
from safetensors.torch import load_file as load_safetensors_file
import json
import argparse
import importlib.util

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint

# This PROJECT_ROOT is for when the script is run directly.
# It will be overridden by the project_root argument if called from elsewhere.
_DEFAULT_PROJECT_ROOT_RUN_TRAINING = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _DEFAULT_PROJECT_ROOT_RUN_TRAINING not in sys.path:
    sys.path.insert(0, _DEFAULT_PROJECT_ROOT_RUN_TRAINING)
    # print(f"DEBUG: run_training.py (direct run) added to sys.path: {_DEFAULT_PROJECT_ROOT_RUN_TRAINING}")

# These imports will work because _DEFAULT_PROJECT_ROOT_RUN_TRAINING (or the one set by callers) is in sys.path
from Model_Training.data_handling.data_loader import get_hf_datasets, load_data_sources_config
from Model_Training.data_handling.collator import multitask_data_collator
from Model_Training.utils.metrics import compute_metrics as project_compute_metrics
from Model_Training.utils.callbacks import PlottingCallback, OnnxExportCallback


def get_landmark_count_from_config(data_sources_path: str, dataset_config_name_key: str) -> int:
    config_data = load_data_sources_config(data_sources_path)
    landmarks = config_data.get("dataset_configurations", {}).get(dataset_config_name_key, {}).get("landmarks")
    if landmarks is None:
        raise ValueError(
            f"Could not find 'landmarks' for '{dataset_config_name_key}' in {data_sources_path}"
        )
    return int(landmarks)


def sanitize_metrics_for_logging(metrics_dict: Dict[str, AnyType]) -> Dict[str, AnyType]:
    sanitized_metrics = {}
    if metrics_dict is None:
        return sanitized_metrics
    for key, value in metrics_dict.items():
        if isinstance(value, list):
            try:
                sanitized_metrics[key] = json.dumps(value)
            except TypeError:
                sanitized_metrics[key] = str(value)
        else:
            sanitized_metrics[key] = value
    return sanitized_metrics


def load_config_module(config_file_path: str) -> AnyType:
    """Dynamically loads a Python module from a given file path."""
    module_name = os.path.splitext(os.path.basename(config_file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, config_file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module at {config_file_path}")

    exp_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp_config_module)
    print(f"Successfully loaded configuration from: {config_file_path}")
    return exp_config_module


def main(exp_config_module: AnyType, project_root: str) -> None:
    """
    Main training function.
    Args:
        exp_config_module: The loaded experiment configuration module.
        project_root: The absolute path to the project root directory (e.g., .../models).
    """
    print("--- Setup ---")
    print(f"Project Root (for module loading): {project_root}")

    # Define the base path for resolving relative paths from config files.
    # This will be .../models/Model_Training/
    model_training_base_path = os.path.join(project_root, "Model_Training")
    print(f"Model Training Base Path (for config relative paths): {model_training_base_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    current_training_hyperparams = exp_config_module.TRAINING_HYPERPARAMS.copy()
    if not torch.cuda.is_available():
        print("CUDA not available. Overriding dataloader_num_workers to 0 and fp16 to False.")
        current_training_hyperparams["dataloader_num_workers"] = 0
        current_training_hyperparams["fp16"] = False
    else:
        current_training_hyperparams["dataloader_num_workers"] = current_training_hyperparams.get(
            "dataloader_num_workers", 4)

    print(f"Dataloader num_workers: {current_training_hyperparams['dataloader_num_workers']}")
    print(f"FP16 training: {current_training_hyperparams['fp16']}")

    print("\n--- Path and Directory Setup ---")

    # Path for data_sources.json is relative to Model_Training/configs
    data_sources_json_path = os.path.join(model_training_base_path, "configs", "data_sources.json")

    # Resolve BASE_OUTPUT_DIR from config relative to model_training_base_path
    base_output_dir_from_config = exp_config_module.BASE_OUTPUT_DIR
    if not os.path.isabs(base_output_dir_from_config):
        # Remove leading "./" or ".\\" if present, then join
        base_output_dir_from_config = os.path.join(model_training_base_path, base_output_dir_from_config.lstrip(".\\/"))

    experiment_run_base_dir = os.path.join(
        base_output_dir_from_config,
        exp_config_module.EXPERIMENT_NAME
    )
    os.makedirs(experiment_run_base_dir, exist_ok=True)

    trainer_artifacts_dir = os.path.join(experiment_run_base_dir, exp_config_module.TRAINER_ARTIFACTS_SUBDIR_NAME)
    os.makedirs(trainer_artifacts_dir, exist_ok=True)

    final_models_output_dir = os.path.join(experiment_run_base_dir, exp_config_module.FINAL_MODELS_SUBDIR_NAME)
    os.makedirs(final_models_output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_output_dir = os.path.join(
        experiment_run_base_dir,
        exp_config_module.PLOTS_PARENT_SUBDIR_NAME,
        f"{timestamp}_{exp_config_module.EXPERIMENT_NAME}_plots"
    )
    os.makedirs(plots_output_dir, exist_ok=True)
    print(f"Data sources JSON path: {data_sources_json_path}")
    print(f"Experiment run base directory: {experiment_run_base_dir}")
    print(f"Plots will be saved in: {plots_output_dir}")

    print("\n--- Loading Datasets ---")
    dataset_config_name_key = getattr(exp_config_module, "DATASET_CONFIG_NAME_KEY",
                                      "EngageNet_10fps_quality95_RandSplit_60_20_20")

    datasets = get_hf_datasets(
        dataset_config_name=dataset_config_name_key,
        data_sources_json_path=data_sources_json_path,
        # This path is now correctly Model_Training/configs/data_sources.json
        train_pipeline=exp_config_module.TRAIN_PIPELINE,
        val_pipeline=exp_config_module.VALIDATION_PIPELINE,
        test_pipeline=exp_config_module.TEST_PIPELINE,
        verbose=True
    )
    train_dataset = datasets.get("train")
    eval_dataset = datasets.get("validation")
    test_dataset = datasets.get("test")

    if not train_dataset or not eval_dataset:
        print("ERROR: Training or Validation dataset failed to load. Exiting.")
        sys.exit(1)
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(eval_dataset)}")
    if test_dataset: print(f"Test dataset size: {len(test_dataset)}")

    print("\n--- Initializing Model ---")
    num_landmarks = get_landmark_count_from_config(data_sources_json_path, dataset_config_name_key)
    model_input_dim = num_landmarks * exp_config_module.NUM_COORDINATES
    model_params_with_input_dim = exp_config_module.MODEL_PARAMS.copy()
    model_params_with_input_dim["input_dim"] = model_input_dim
    model = exp_config_module.MODEL_CLASS(
        **model_params_with_input_dim,
        regression_loss_fn=exp_config_module.REGRESSION_LOSS_FN,
        classification_loss_fn=exp_config_module.CLASSIFICATION_LOSS_FN
    )

    if hasattr(exp_config_module, 'LOAD_INITIAL_WEIGHTS_PATH') and exp_config_module.LOAD_INITIAL_WEIGHTS_PATH:
        load_path_from_config = exp_config_module.LOAD_INITIAL_WEIGHTS_PATH
        # Resolve load_path relative to model_training_base_path if it's a relative path
        if not os.path.isabs(load_path_from_config):
            load_path = os.path.join(model_training_base_path, load_path_from_config.lstrip(".\\/"))
        else:
            load_path = load_path_from_config  # It's an absolute path

        print(f"Attempting to load initial weights from: {load_path}")
        if os.path.exists(load_path):
            try:
                map_location_for_load = torch.device('cpu')
                if load_path.endswith(".safetensors"):
                    state_dict = load_safetensors_file(load_path, device=map_location_for_load.type)
                elif load_path.endswith((".bin", ".pth")):
                    state_dict = torch.load(load_path, map_location=map_location_for_load)
                else:
                    state_dict = torch.load(load_path, map_location=map_location_for_load)
                if isinstance(state_dict, dict) and all(key.startswith("module.") for key in state_dict.keys()):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=True)
                print(f"Successfully loaded initial weights from {load_path}")
            except Exception as e:
                print(f"Error loading initial weights from {load_path}: {e}. Starting with random weights.")
        else:
            print(f"Warning: LOAD_INITIAL_WEIGHTS_PATH '{load_path}' not found. Starting with random weights.")
    print(f"Model '{exp_config_module.MODEL_CLASS.__name__}' initialized.")

    print("\n--- Defining Training Arguments ---")
    # output_dir for TrainingArguments should be an absolute path.
    # trainer_artifacts_dir is already constructed based on experiment_run_base_dir, which is now correctly rooted.
    training_args = TrainingArguments(
        output_dir=trainer_artifacts_dir,  # This is now an absolute path
        **current_training_hyperparams
    )
    print(f"Effective device for Trainer: {training_args.device}")
    print(f"Trainer output_dir: {training_args.output_dir}")

    compute_metrics_fn = functools.partial(
        project_compute_metrics,
        idx_to_score_map=exp_config_module.IDX_TO_SCORE_MAP,
        idx_to_name_map=exp_config_module.IDX_TO_NAME_MAP,
        num_classes_classification=exp_config_module.NUM_CLASSES_CLASSIFICATION
    )

    print("\n--- Setting up Callbacks ---")
    callbacks = []
    if hasattr(exp_config_module, 'EARLY_STOPPING_PATIENCE') and \
            hasattr(exp_config_module, 'EARLY_STOPPING_THRESHOLD'):
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=exp_config_module.EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=exp_config_module.EARLY_STOPPING_THRESHOLD
        ))

    if hasattr(exp_config_module, 'PLOTTING_CALLBACK_PARAMS') and \
            hasattr(exp_config_module, 'IDX_TO_NAME_MAP'):
        plotting_cb_params = exp_config_module.PLOTTING_CALLBACK_PARAMS.copy()
        # plots_output_dir is already an absolute path, correctly rooted.
        plot_callback = PlottingCallback(
            plots_save_dir=plots_output_dir,
            idx_to_name_map=exp_config_module.IDX_TO_NAME_MAP,
            loss_plot_filename=plotting_cb_params.get("loss_plot_filename", "loss_curves.png"),
            lr_plot_filename=plotting_cb_params.get("lr_plot_filename", "learning_rate_curve.png"),
            regression_metrics_plot_filename=plotting_cb_params.get("regression_metrics_plot_filename",
                                                                    "regression_metrics.png"),
            classification_metrics_plot_filename=plotting_cb_params.get("classification_metrics_plot_filename",
                                                                        "classification_metrics.png"),
            confusion_matrix_eval_filename=plotting_cb_params.get("confusion_matrix_eval_filename",
                                                                  "confusion_matrix_eval.png"),
            confusion_matrix_test_filename=plotting_cb_params.get("confusion_matrix_test_filename",
                                                                  "confusion_matrix_test.png")
        )
        callbacks.append(plot_callback)
    print(f"PlottingCallback configured (if parameters were present). Plots will be saved to: {plots_output_dir}")

    if hasattr(exp_config_module, 'PERFORM_ONNX_EXPORT') and exp_config_module.PERFORM_ONNX_EXPORT:
        if hasattr(exp_config_module, 'ONNX_EXPORT_PARAMS'):
            onnx_params_from_config = exp_config_module.ONNX_EXPORT_PARAMS.copy()
            onnx_model_filename = onnx_params_from_config.pop("onnx_model_filename", "model.onnx")
            # final_models_output_dir is already an absolute path, correctly rooted.
            onnx_model_save_path = os.path.join(
                final_models_output_dir,
                onnx_model_filename
            )
            seq_len_onnx = onnx_params_from_config.get("representative_seq_len", 30)
            input_shape_for_onnx = (seq_len_onnx, num_landmarks, exp_config_module.NUM_COORDINATES)
            onnx_callback_args = {
                "onnx_model_full_save_path": onnx_model_save_path,
                "input_shape_for_onnx": input_shape_for_onnx,
                "opset_version": onnx_params_from_config.get("opset_version"),
                "onnx_input_names": onnx_params_from_config.get("input_names"),
                "onnx_output_names": onnx_params_from_config.get("output_names"),
                "onnx_dynamic_axes": onnx_params_from_config.get("dynamic_axes")
            }
            onnx_callback_args_filtered = {k: v for k, v in onnx_callback_args.items() if v is not None}
            callbacks.append(OnnxExportCallback(**onnx_callback_args_filtered))
            print(f"OnnxExportCallback configured. ONNX model will be saved to: {onnx_model_save_path}")
        else:
            print("PERFORM_ONNX_EXPORT is True, but ONNX_EXPORT_PARAMS not found in config. Skipping ONNX export.")
    else:
        print("ONNX export is disabled or PERFORM_ONNX_EXPORT not found in config.")

    print("\n--- Initializing Trainer ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=multitask_data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )
    print("Trainer initialized.")

    print("\n--- Starting Training ---")
    resume_from_checkpoint_path_config = None
    if hasattr(exp_config_module, 'RESUME_FROM_CHECKPOINT'):
        resume_from_checkpoint_path_config = exp_config_module.RESUME_FROM_CHECKPOINT

    actual_resume_path: Optional[Union[str, bool]] = None
    if resume_from_checkpoint_path_config is True:
        last_checkpoint = get_last_checkpoint(trainer_artifacts_dir)  # trainer_artifacts_dir is absolute
        if last_checkpoint:
            actual_resume_path = last_checkpoint
            print(f"Resuming from last checkpoint: {actual_resume_path}")
        else:
            print(f"No checkpoint in {trainer_artifacts_dir} to resume from (RESUME_FROM_CHECKPOINT was True).")
    elif isinstance(resume_from_checkpoint_path_config, str):
        # Resolve relative path from config to be relative to model_training_base_path
        if not os.path.isabs(resume_from_checkpoint_path_config):
            # Check if it's a simple name (potential last_checkpoint folder name) vs a relative path
            # If it contains path separators, treat it as a path relative to model_training_base_path
            # Otherwise, it might be a checkpoint folder name relative to trainer_artifacts_dir (Hugging Face default)
            if os.path.sep in resume_from_checkpoint_path_config or \
                    ('/' in resume_from_checkpoint_path_config or '\\' in resume_from_checkpoint_path_config):
                actual_resume_path = os.path.join(model_training_base_path,
                                                  resume_from_checkpoint_path_config.lstrip(".\\/"))
            else:  # It's a simple name, assume it's relative to trainer_artifacts_dir
                actual_resume_path = os.path.join(trainer_artifacts_dir, resume_from_checkpoint_path_config)
        else:  # It's an absolute path
            actual_resume_path = resume_from_checkpoint_path_config
        print(f"Attempting to resume from specified checkpoint: {actual_resume_path}")
    # If resume_from_checkpoint_path_config is False or None, actual_resume_path remains None

    if not actual_resume_path:
        print("Starting fresh training run (no valid resume path determined).")

    try:
        train_result = trainer.train(resume_from_checkpoint=actual_resume_path)
        print("Training finished.")
        if train_result and train_result.metrics:
            print(f"Train Output Metrics (raw): {train_result.metrics}")
            sanitized_train_metrics = sanitize_metrics_for_logging(train_result.metrics)
            trainer.log_metrics("train", sanitized_train_metrics)
            trainer.save_metrics("train", sanitized_train_metrics)
        trainer.save_state()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if hasattr(exp_config_module, 'SAVE_FINAL_PYTORCH_MODEL') and exp_config_module.SAVE_FINAL_PYTORCH_MODEL:
        # final_models_output_dir is already an absolute path
        print(f"\n--- Saving Final Best PyTorch Model to: {final_models_output_dir} ---")
        try:
            trainer.save_model(final_models_output_dir)
            print(f"Best PyTorch model saved to {final_models_output_dir}")
        except Exception as e:
            print(f"Error saving final PyTorch model: {e}")
    else:
        print("\nSkipping save of final PyTorch model (or SAVE_FINAL_PYTORCH_MODEL not True in config).")

    print("\n--- Checking model device post-train/load_best_model ---")
    actual_model_for_eval = trainer.model
    unwrapped_model_for_eval = actual_model_for_eval
    while hasattr(unwrapped_model_for_eval, 'module'):
        unwrapped_model_for_eval = unwrapped_model_for_eval.module

    first_param_device = next(unwrapped_model_for_eval.parameters()).device
    print(f"Device of the first parameter of the unwrapped model: {first_param_device}")
    if first_param_device != training_args.device:
        print(
            f"Warning: Model device ({first_param_device}) might not match trainer device ({training_args.device}) before evaluation.")

    print("\n--- Evaluating Best Model on Validation Set ---")
    if hasattr(trainer, 'model') and trainer.model is not None:
        trainer.model.to(training_args.device)

    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    print(f"Validation Set Evaluation Results (raw): {eval_results}")
    sanitized_eval_results = sanitize_metrics_for_logging(eval_results)
    trainer.log_metrics("eval_best", sanitized_eval_results)
    try:
        trainer.save_metrics("eval_best", eval_results)
    except TypeError:
        print("Warning: Failed to save original eval_results with lists. Saving sanitized version.")
        trainer.save_metrics("eval_best", sanitized_eval_results)

    if test_dataset:
        print("\n--- Evaluating Best Model on Test Set ---")
        if hasattr(trainer, 'model') and trainer.model is not None:
            trainer.model.to(training_args.device)
        test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        print(f"Test Set Evaluation Results (raw): {test_results}")
        sanitized_test_results = sanitize_metrics_for_logging(test_results)
        trainer.log_metrics("test_best", sanitized_test_results)
        try:
            trainer.save_metrics("test_best", test_results)
        except TypeError:
            print("Warning: Failed to save original test_results with lists. Saving sanitized version.")
            trainer.save_metrics("test_best", sanitized_test_results)
    else:
        print("Test dataset not available. Skipping test set evaluation.")

    print("\n--- Script Finished ---")


if __name__ == "__main__":
    print(f"DEBUG: Running {__file__} as __main__")

    parser = argparse.ArgumentParser(description="Run training with a specified configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment configuration Python file."
    )
    args = parser.parse_args()

    current_project_root = _DEFAULT_PROJECT_ROOT_RUN_TRAINING

    if os.path.isabs(args.config):
        config_file_path_resolved = args.config
    else:
        # If config path is relative, assume it's relative to the project_root (e.g., .../models)
        # Example: --config Model_Training/configs/my_config.py
        config_file_path_resolved = os.path.join(current_project_root, args.config)

    if not os.path.exists(config_file_path_resolved):
        config_file_path_cwd = os.path.abspath(args.config)
        if os.path.exists(config_file_path_cwd):
            config_file_path_resolved = config_file_path_cwd
            print(
                f"DEBUG: Config path '{args.config}' not found relative to project root, but found relative to CWD: {config_file_path_resolved}")
        else:
            print(
                f"Error: Configuration file not found at '{args.config}' (tried absolute, relative to project root '{current_project_root}', and relative to CWD).")
            sys.exit(1)

    try:
        exp_config_module_loaded = load_config_module(config_file_path_resolved)
    except Exception as e:
        print(f"Error loading configuration module from {config_file_path_resolved}: {e}")
        sys.exit(1)

    main(exp_config_module=exp_config_module_loaded, project_root=current_project_root)
