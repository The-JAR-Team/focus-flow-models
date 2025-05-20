import os
import sys
import torch
# import json # Not directly used in this snippet, but likely in full file
import numpy as np
import functools
from datetime import datetime
from typing import Optional, Dict, Any, Union, Mapping  # Added Mapping
from safetensors.torch import load_file as load_safetensors_file
import json  # For converting list to string safely

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Model_Training.configs import experiment_config_v4 as exp_config
from Model_Training.data_handling.data_loader import get_hf_datasets, load_data_sources_config
from Model_Training.data_handling.collator import multitask_data_collator
# Model is imported via exp_config.MODEL_CLASS

from Model_Training.utils.metrics import compute_metrics as project_compute_metrics
from Model_Training.utils.callbacks import PlottingCallback, OnnxExportCallback  # Ensure this is the updated callback


# from Model_Training.utils.onnx_exporter import export_to_onnx # Used by OnnxExportCallback


def get_landmark_count_from_config(data_sources_path: str, dataset_config_name_key: str) -> int:
    config_data = load_data_sources_config(data_sources_path)
    landmarks = config_data.get("dataset_configurations", {}).get(dataset_config_name_key, {}).get("landmarks")
    if landmarks is None:
        raise ValueError(
            f"Could not find 'landmarks' for '{dataset_config_name_key}' in {data_sources_path}"
        )
    return int(landmarks)


# Helper function to sanitize metrics for console logging and JSON saving
def sanitize_metrics_for_logging(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts list-based metrics to strings to prevent formatting errors during logging.
    """
    sanitized_metrics = {}
    if metrics_dict is None:
        return sanitized_metrics
    for key, value in metrics_dict.items():
        if isinstance(value, list):
            # Convert list to a string representation, e.g., using json.dumps or str()
            # Using str() is simpler for console, json.dumps for more robust string representation
            try:
                sanitized_metrics[key] = json.dumps(value)  # More robust for nested lists
            except TypeError:
                sanitized_metrics[key] = str(value)  # Fallback
            # Or, you could use a placeholder:
            # sanitized_metrics[key] = f"[List data for {key}]"
        else:
            sanitized_metrics[key] = value
    return sanitized_metrics


def main(project_root: str = PROJECT_ROOT) -> None:
    print("--- Setup ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    current_training_hyperparams = exp_config.TRAINING_HYPERPARAMS.copy()
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
    data_sources_json_path = os.path.join(project_root, "configs", "data_sources.json")
    experiment_run_base_dir = os.path.join(
        project_root,
        exp_config.BASE_OUTPUT_DIR.lstrip("./\\"),
        exp_config.EXPERIMENT_NAME
    )
    os.makedirs(experiment_run_base_dir, exist_ok=True)
    trainer_artifacts_dir = os.path.join(experiment_run_base_dir, exp_config.TRAINER_ARTIFACTS_SUBDIR_NAME)
    os.makedirs(trainer_artifacts_dir, exist_ok=True)
    final_models_output_dir = os.path.join(experiment_run_base_dir, exp_config.FINAL_MODELS_SUBDIR_NAME)
    os.makedirs(final_models_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_output_dir = os.path.join(
        experiment_run_base_dir,
        exp_config.PLOTS_PARENT_SUBDIR_NAME,
        f"{timestamp}_{exp_config.EXPERIMENT_NAME}_plots"
    )
    os.makedirs(plots_output_dir, exist_ok=True)
    print(f"Plots will be saved in: {plots_output_dir}")

    print("\n--- Loading Datasets ---")
    dataset_config_name_key = "EngageNet_10fps_quality95_RandSplit_60_20_20"  # Example
    datasets = get_hf_datasets(
        dataset_config_name=dataset_config_name_key,
        data_sources_json_path=data_sources_json_path,
        train_pipeline=exp_config.TRAIN_PIPELINE,
        val_pipeline=exp_config.VALIDATION_PIPELINE,
        test_pipeline=exp_config.TEST_PIPELINE,
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
    model_input_dim = num_landmarks * exp_config.NUM_COORDINATES
    model_params_with_input_dim = exp_config.MODEL_PARAMS.copy()
    model_params_with_input_dim["input_dim"] = model_input_dim
    model = exp_config.MODEL_CLASS(
        **model_params_with_input_dim,
        regression_loss_fn=exp_config.REGRESSION_LOSS_FN,
        classification_loss_fn=exp_config.CLASSIFICATION_LOSS_FN
    )

    if exp_config.LOAD_INITIAL_WEIGHTS_PATH:
        load_path = exp_config.LOAD_INITIAL_WEIGHTS_PATH
        if not os.path.isabs(load_path) and load_path.startswith("./"):
            load_path = os.path.join(project_root, load_path.lstrip("./\\"))
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
    print(f"Model '{exp_config.MODEL_CLASS.__name__}' initialized.")

    print("\n--- Defining Training Arguments ---")
    training_args = TrainingArguments(
        output_dir=trainer_artifacts_dir,
        **current_training_hyperparams
    )
    print(f"Effective device for Trainer: {training_args.device}")

    compute_metrics_fn = functools.partial(
        project_compute_metrics,
        idx_to_score_map=exp_config.IDX_TO_SCORE_MAP,
        idx_to_name_map=exp_config.IDX_TO_NAME_MAP,
        num_classes_classification=exp_config.NUM_CLASSES_CLASSIFICATION
    )

    print("\n--- Setting up Callbacks ---")
    callbacks = []
    callbacks.append(EarlyStoppingCallback(
        early_stopping_patience=exp_config.EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=exp_config.EARLY_STOPPING_THRESHOLD
    ))

    plotting_cb_params = exp_config.PLOTTING_CALLBACK_PARAMS.copy()
    plot_callback = PlottingCallback(  # Ensure this is the updated PlottingCallback
        plots_save_dir=plots_output_dir,
        idx_to_name_map=exp_config.IDX_TO_NAME_MAP,
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
    print(f"PlottingCallback configured. Plots will be saved to: {plots_output_dir}")

    if exp_config.PERFORM_ONNX_EXPORT:
        onnx_params_from_config = exp_config.ONNX_EXPORT_PARAMS.copy()
        onnx_model_save_path = os.path.join(
            final_models_output_dir,
            onnx_params_from_config.pop("onnx_model_filename")
        )
        seq_len_onnx = onnx_params_from_config.get("representative_seq_len", 30)
        input_shape_for_onnx = (seq_len_onnx, num_landmarks, exp_config.NUM_COORDINATES)
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
        print("ONNX export is disabled.")

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
    resume_from_checkpoint_path: Optional[Union[str, bool]] = None
    if hasattr(exp_config, 'RESUME_FROM_CHECKPOINT'):
        resume_from_checkpoint_path = exp_config.RESUME_FROM_CHECKPOINT
        if resume_from_checkpoint_path is True:
            last_checkpoint = get_last_checkpoint(trainer_artifacts_dir)
            if last_checkpoint:
                resume_from_checkpoint_path = last_checkpoint
            else:
                resume_from_checkpoint_path = None; print(f"No checkpoint in {trainer_artifacts_dir} to resume from.")
        elif isinstance(resume_from_checkpoint_path, str):
            print(f"Attempting to resume from: {resume_from_checkpoint_path}")
        else:
            resume_from_checkpoint_path = None
    if not resume_from_checkpoint_path: print("Starting fresh training run.")

    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint_path)
        print("Training finished.")
        if train_result and train_result.metrics:
            print(f"Train Output Metrics (raw): {train_result.metrics}")
            sanitized_train_metrics = sanitize_metrics_for_logging(train_result.metrics)
            trainer.log_metrics("train", sanitized_train_metrics)
            trainer.save_metrics("train",
                                 sanitized_train_metrics)  # train_result.metrics might be better if JSON handles lists
        trainer.save_state()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if exp_config.SAVE_FINAL_PYTORCH_MODEL:
        print(f"\n--- Saving Final Best PyTorch Model to: {final_models_output_dir} ---")
        try:
            trainer.save_model(final_models_output_dir)
            print(f"Best PyTorch model saved to {final_models_output_dir}")
        except Exception as e:
            print(f"Error saving final PyTorch model: {e}")
    else:
        print("\nSkipping save of final PyTorch model.")

    # --- Debugging Device Mismatch before final evaluation (as in original) ---
    # (Keeping your existing device check logic here)
    print("\n--- Checking model device post-train/load_best_model ---")
    actual_model_for_eval = trainer.model
    unwrapped_model_for_eval = actual_model_for_eval
    while hasattr(unwrapped_model_for_eval, 'module'):
        unwrapped_model_for_eval = unwrapped_model_for_eval.module
    if hasattr(unwrapped_model_for_eval, 'frame_norm'):
        # ... (your device check logic) ...
        pass  # Placeholder for brevity
    else:
        print("Could not find 'frame_norm' attribute on the unwrapped model.")

    print("\n--- Evaluating Best Model on Validation Set ---")
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    # eval_results now contains the raw metrics, including lists for confusion matrices
    print(f"Validation Set Evaluation Results (raw): {eval_results}")

    # Sanitize metrics before logging to console or saving as JSON that might not support lists directly
    # The PlottingCallback will use the original eval_results from state.log_history
    sanitized_eval_results = sanitize_metrics_for_logging(eval_results)
    trainer.log_metrics("eval_best", sanitized_eval_results)  # This will prevent the TypeError

    # For save_metrics, Hugging Face's default implementation usually handles
    # JSON serialization of basic list/dict structures well.
    # However, if it causes issues, you can also use the sanitized version.
    # Let's assume the default save_metrics can handle it for now,
    # as it's mainly the console print formatting that breaks.
    # If save_metrics also breaks, use: trainer.save_metrics("eval_best", sanitized_eval_results)
    try:
        trainer.save_metrics("eval_best", eval_results)  # Try with original first for full data in JSON
    except TypeError:
        print("Warning: Failed to save original eval_results with lists. Saving sanitized version.")
        trainer.save_metrics("eval_best", sanitized_eval_results)

    if test_dataset:
        print("\n--- Evaluating Best Model on Test Set ---")
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
    main()
