# Model_Training/main/run_training.py
import os
import sys
import torch
import json
import numpy as np
import functools  # For partial function application
from datetime import datetime
from typing import Optional, Dict, Any, Union  # For type hints

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint  # For resuming

# --- Add project root to Python path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Project-specific imports ---
from Model_Training.configs import experiment_config_v4 as exp_config
from Model_Training.data_handling.data_loader import get_hf_datasets, load_data_sources_config
from Model_Training.data_handling.collator import multitask_data_collator
# Model_Training.models.multitask_gru_attention_model_v4 is imported via exp_config.MODEL_CLASS

# Import utility functions and callbacks
from Model_Training.utils.metrics import compute_metrics as project_compute_metrics  # Renamed to avoid conflict
from Model_Training.utils.callbacks import PlottingCallback, OnnxExportCallback
from Model_Training.utils.onnx_exporter import export_to_onnx  # Though OnnxExportCallback uses this


# --- Helper to load landmark count ---
def get_landmark_count_from_config(data_sources_path: str, dataset_config_name_key: str) -> int:
    config_data = load_data_sources_config(data_sources_path)
    landmarks = config_data.get("dataset_configurations", {}).get(dataset_config_name_key, {}).get("landmarks")
    if landmarks is None:
        raise ValueError(
            f"Could not find 'landmarks' for '{dataset_config_name_key}' in {data_sources_path}"
        )
    return int(landmarks)


def main():
    # --- 0. Determine Device and Dataloader Workers ---
    print("--- Setup ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Adjust dataloader_num_workers based on CUDA availability and config
    if torch.cuda.is_available():
        dataloader_num_workers = exp_config.TRAINING_HYPERPARAMS.get("dataloader_num_workers", 4)
    else:
        print("CUDA not available. Setting dataloader_num_workers to 1 (or 0 if preferred for CPU).")
        # Often 0 or 1 is better for CPU to avoid overhead, but let's use 1 as a safe default.
        # If your config specifies 0 for CPU, that would also be fine.
        # Forcing to 1 if CUDA is not available, overriding config if it's higher.
        # dataloader_num_workers = min(exp_config.TRAINING_HYPERPARAMS.get("dataloader_num_workers", 1), 1)
        # If you strictly want 0 for CPU, uncomment below:
        dataloader_num_workers = 0

    print(f"Dataloader num_workers: {dataloader_num_workers}")
    # Update training hyperparams with the determined num_workers
    current_training_hyperparams = exp_config.TRAINING_HYPERPARAMS.copy()
    current_training_hyperparams["dataloader_num_workers"] = dataloader_num_workers
    if not torch.cuda.is_available() and current_training_hyperparams.get("fp16"):
        print("CUDA not available. Disabling fp16 training.")
        current_training_hyperparams["fp16"] = False

    # --- 1. Paths and Directory Setup ---
    print("\n--- Path and Directory Setup ---")
    data_sources_json_path = os.path.join(PROJECT_ROOT, "configs", "data_sources.json")

    # Base path for all outputs of this specific experiment run
    experiment_run_base_dir = os.path.join(
        PROJECT_ROOT,
        exp_config.BASE_OUTPUT_DIR.lstrip("./\\"),  # Remove potential leading ./
        exp_config.EXPERIMENT_NAME
    )
    os.makedirs(experiment_run_base_dir, exist_ok=True)
    print(f"Base directory for this experiment run: {experiment_run_base_dir}")

    # Directory for Hugging Face Trainer artifacts (checkpoints, logs, etc.)
    trainer_artifacts_dir = os.path.join(experiment_run_base_dir, exp_config.TRAINER_ARTIFACTS_SUBDIR_NAME)
    os.makedirs(trainer_artifacts_dir, exist_ok=True)
    print(f"Trainer artifacts (checkpoints, logs) will be in: {trainer_artifacts_dir}")

    # Directory for final exported models (PyTorch .bin, ONNX .onnx)
    final_models_output_dir = os.path.join(experiment_run_base_dir, exp_config.FINAL_MODELS_SUBDIR_NAME)
    os.makedirs(final_models_output_dir, exist_ok=True)
    print(f"Final exported models (PyTorch, ONNX) will be in: {final_models_output_dir}")

    # Directory for plots (timestamped)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir_name = f"{timestamp}_{exp_config.EXPERIMENT_NAME}_plots"
    plots_output_dir = os.path.join(experiment_run_base_dir, exp_config.PLOTS_PARENT_SUBDIR_NAME, plots_dir_name)
    # The PlottingCallback will create this directory if it doesn't exist.
    print(
        f"Plots will be saved in a subfolder within: {os.path.join(experiment_run_base_dir, exp_config.PLOTS_PARENT_SUBDIR_NAME)}")

    # --- 2. Load Datasets ---
    print("\n--- Loading Datasets ---")
    # This could be a script argument in a more advanced setup
    dataset_config_name_key = "EngageNet_10fps_quality95_RandSplit_60_20_20"  # From your data_sources.json

    datasets = get_hf_datasets(
        dataset_config_name=dataset_config_name_key,
        data_sources_json_path=data_sources_json_path,
        train_pipeline=exp_config.TRAIN_PIPELINE,
        val_pipeline=exp_config.VALIDATION_PIPELINE,
        test_pipeline=exp_config.TEST_PIPELINE,
        verbose=True  # Or make this configurable
    )
    train_dataset = datasets.get("train")
    eval_dataset = datasets.get("validation")
    test_dataset = datasets.get("test")

    if not train_dataset:
        print("ERROR: Training dataset failed to load. Exiting.")
        sys.exit(1)
    if not eval_dataset:
        print("ERROR: Validation dataset failed to load. Exiting.")
        sys.exit(1)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")
    if test_dataset:
        print(f"Test dataset size: {len(test_dataset)}")

    # --- 3. Initialize Model ---
    print("\n--- Initializing Model ---")
    num_landmarks = get_landmark_count_from_config(data_sources_json_path, dataset_config_name_key)
    model_input_dim = num_landmarks * exp_config.NUM_COORDINATES
    print(
        f"Determined model input dimension: {model_input_dim} ({num_landmarks} landmarks * {exp_config.NUM_COORDINATES} coords)")

    model_params_with_input_dim = exp_config.MODEL_PARAMS.copy()
    model_params_with_input_dim["input_dim"] = model_input_dim

    model = exp_config.MODEL_CLASS(
        **model_params_with_input_dim,
        regression_loss_fn=exp_config.REGRESSION_LOSS_FN,
        classification_loss_fn=exp_config.CLASSIFICATION_LOSS_FN
    )

    # Load initial weights if specified (and not resuming a full trainer checkpoint)
    # Note: RESUME_FROM_CHECKPOINT logic is handled by Trainer.train() argument later.
    # This section is for starting a *new* training run with pre-initialized weights.
    if exp_config.LOAD_INITIAL_WEIGHTS_PATH:
        print(f"Attempting to load initial weights from: {exp_config.LOAD_INITIAL_WEIGHTS_PATH}")
        try:
            if os.path.exists(exp_config.LOAD_INITIAL_WEIGHTS_PATH):
                state_dict = torch.load(exp_config.LOAD_INITIAL_WEIGHTS_PATH, map_location=device)
                # Handle potential "module." prefix if saved from DataParallel/DDP
                if isinstance(state_dict, dict) and all(key.startswith("module.") for key in state_dict.keys()):
                    print("Removing 'module.' prefix from state_dict keys.")
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

                model.load_state_dict(state_dict, strict=True)  # Set strict=False if you expect missing/extra keys
                print(f"Successfully loaded initial weights into model from {exp_config.LOAD_INITIAL_WEIGHTS_PATH}")
            else:
                print(
                    f"Warning: LOAD_INITIAL_WEIGHTS_PATH '{exp_config.LOAD_INITIAL_WEIGHTS_PATH}' not found. Starting with random weights.")
        except Exception as e:
            print(
                f"Error loading initial weights from {exp_config.LOAD_INITIAL_WEIGHTS_PATH}: {e}. Starting with random weights.")

    model.to(device)
    print(f"Model '{exp_config.MODEL_CLASS.__name__}' initialized and moved to {device}.")

    # --- 4. Define Training Arguments ---
    print("\n--- Defining Training Arguments ---")
    training_args = TrainingArguments(
        output_dir=trainer_artifacts_dir,  # Trainer saves its checkpoints and logs here
        **current_training_hyperparams  # Unpack from config
    )
    print(f"Training arguments defined. Trainer output (checkpoints, logs) will be in: {training_args.output_dir}")

    # --- 5. Define Compute Metrics Function ---
    # Use functools.partial to pass additional arguments from exp_config to your compute_metrics
    compute_metrics_fn = functools.partial(
        project_compute_metrics,  # Your function from Model_Training.utils.metrics
        idx_to_score_map=exp_config.IDX_TO_SCORE_MAP,
        idx_to_name_map=exp_config.IDX_TO_NAME_MAP,
        num_classes_classification=exp_config.NUM_CLASSES_CLASSIFICATION
    )

    # --- 6. Callbacks ---
    print("\n--- Setting up Callbacks ---")
    callbacks = []
    callbacks.append(EarlyStoppingCallback(
        early_stopping_patience=exp_config.EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=exp_config.EARLY_STOPPING_THRESHOLD
    ))

    # Plotting Callback
    # Ensure plots_output_dir is created by the callback or here
    os.makedirs(plots_output_dir, exist_ok=True)
    plotting_cb_params = exp_config.PLOTTING_CALLBACK_PARAMS.copy()
    plotting_cb_params['plots_save_dir'] = plots_output_dir  # Pass the specific save directory

    # Check if your PlottingCallback has been updated to accept plots_save_dir
    try:
        plot_callback = PlottingCallback(**plotting_cb_params)
        callbacks.append(plot_callback)
        print(f"PlottingCallback configured. Plots will be saved to: {plots_output_dir}")
    except TypeError as e:
        print(f"Warning: Could not initialize PlottingCallback with 'plots_save_dir'. Error: {e}")
        print("Attempting to initialize PlottingCallback without 'plots_save_dir'. Plots may go to trainer output dir.")
        # Fallback if PlottingCallback wasn't updated
        simple_plotting_params = {k: v for k, v in exp_config.PLOTTING_CALLBACK_PARAMS.items() if k != 'plots_save_dir'}
        try:
            callbacks.append(PlottingCallback(**simple_plotting_params))
        except Exception as e_simple:
            print(f"Failed to initialize PlottingCallback even in simplified form: {e_simple}")

    # ONNX Export Callback (conditionally added)
    if exp_config.PERFORM_ONNX_EXPORT:
        onnx_params_from_config = exp_config.ONNX_EXPORT_PARAMS.copy()

        # Construct full path for the ONNX model
        onnx_model_save_path = os.path.join(
            final_models_output_dir,
            onnx_params_from_config.pop("onnx_model_filename")  # Get filename and remove from dict
        )

        # Prepare input_shape_for_onnx: (seq_len, num_landmarks, num_coords)
        # num_landmarks and num_coords are known. seq_len from config.
        seq_len_onnx = onnx_params_from_config.get("representative_seq_len", 30)
        input_shape_for_onnx = (seq_len_onnx, num_landmarks, exp_config.NUM_COORDINATES)

        # These are args for OnnxExportCallback's __init__
        onnx_callback_args = {
            "onnx_model_full_save_path": onnx_model_save_path,
            "input_shape_for_onnx": input_shape_for_onnx,
            "opset_version": onnx_params_from_config.get("opset_version"),
            "onnx_input_names": onnx_params_from_config.get("input_names"),
            "onnx_output_names": onnx_params_from_config.get("output_names"),
            "onnx_dynamic_axes": onnx_params_from_config.get("dynamic_axes")
        }
        # Filter out None values before passing to callback
        onnx_callback_args_filtered = {k: v for k, v in onnx_callback_args.items() if v is not None}

        try:
            onnx_export_callback = OnnxExportCallback(**onnx_callback_args_filtered)
            callbacks.append(onnx_export_callback)
            print(f"OnnxExportCallback configured. ONNX model will be saved to: {onnx_model_save_path}")
        except TypeError as e:
            print(f"Warning: Could not initialize OnnxExportCallback with 'onnx_model_full_save_path'. Error: {e}")
            print("ONNX export might not work as expected or save to trainer's output dir.")
            # Fallback logic might be needed if callback signature is different.
        except Exception as e_onnx_init:
            print(f"Failed to initialize OnnxExportCallback: {e_onnx_init}")
    else:
        print("ONNX export is disabled via config (PERFORM_ONNX_EXPORT=False).")

    # --- 7. Initialize Trainer ---
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

    # --- 8. Start Training ---
    print("\n--- Starting Training ---")
    # Handle resuming from checkpoint if specified.
    # Note: User prefers LOAD_INITIAL_WEIGHTS_PATH for starting, so RESUME_FROM_CHECKPOINT might be None.
    # If it were set to True or a path, Trainer handles it.
    # We will let `trainer.train` handle the resume_from_checkpoint logic as it is designed for it.
    # The `TrainingArguments` will be the primary source for finding the last checkpoint if `output_dir`
    # already contains checkpoints and `overwrite_output_dir=False` (default).
    # Or, explicitly:
    resume_from_checkpoint_path: Optional[Union[str, bool]] = None
    if hasattr(exp_config, 'RESUME_FROM_CHECKPOINT'):  # Check if the attribute exists
        resume_from_checkpoint_path = exp_config.RESUME_FROM_CHECKPOINT
        if resume_from_checkpoint_path is True:
            # Attempt to find the last checkpoint in the trainer_artifacts_dir
            last_checkpoint = get_last_checkpoint(trainer_artifacts_dir)
            if last_checkpoint:
                print(f"Resuming training from last checkpoint: {last_checkpoint}")
                resume_from_checkpoint_path = last_checkpoint
            else:
                print(
                    f"RESUME_FROM_CHECKPOINT was True, but no checkpoint found in {trainer_artifacts_dir}. Starting fresh.")
                resume_from_checkpoint_path = None  # Start fresh
        elif isinstance(resume_from_checkpoint_path, str):
            print(f"Attempting to resume training from specified checkpoint: {resume_from_checkpoint_path}")
        else:  # None or False
            resume_from_checkpoint_path = None  # Start fresh
            print("Starting fresh training run (no checkpoint specified for resuming).")
    else:  # If RESUME_FROM_CHECKPOINT is not in config, start fresh
        print("RESUME_FROM_CHECKPOINT not defined in config. Starting fresh training run.")

    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint_path)
        print("Training finished.")
        # train_result might be None if training is interrupted early or resume_from_checkpoint is tricky
        if train_result:
            print(f"Train Output Metrics: {train_result.metrics}")
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)  # Saves to trainer_artifacts_dir

        # Save final state (not just model) which might be useful
        trainer.save_state()  # Saves optimizer, scheduler, RNG states to trainer_artifacts_dir

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  # Exit if training fails

    # --- 9. Save Final Best PyTorch Model (if enabled) ---
    # Trainer with load_best_model_at_end=True already has the best model in trainer.model
    if exp_config.SAVE_FINAL_PYTORCH_MODEL:
        print(f"\n--- Saving Final Best PyTorch Model to: {final_models_output_dir} ---")
        try:
            # This saves pytorch_model.bin, config.json, training_args.bin etc.
            trainer.save_model(final_models_output_dir)
            print(f"Best PyTorch model saved successfully to {final_models_output_dir}")
        except Exception as e:
            print(f"Error saving final PyTorch model: {e}")
    else:
        print("\nSkipping save of final best PyTorch model (SAVE_FINAL_PYTORCH_MODEL=False).")

    # Note: ONNX export is handled by the OnnxExportCallback if PERFORM_ONNX_EXPORT was True.
    # The callback acts on the best model loaded by the Trainer at on_train_end.

    # --- 10. Evaluate on Validation & Test Set (using the best model) ---
    print("\n--- Evaluating Best Model on Validation Set ---")
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    print(f"Validation Set Evaluation Results (Best Model): {eval_results}")
    trainer.log_metrics("eval_best", eval_results)  # Log with a distinct prefix
    trainer.save_metrics("eval_best", eval_results)  # Saves to trainer_artifacts_dir

    if test_dataset:
        print("\n--- Evaluating Best Model on Test Set ---")
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        print(f"Test Set Evaluation Results (Best Model): {test_results}")
        trainer.log_metrics("test_best", test_results)
        trainer.save_metrics("test_best", test_results)  # Saves to trainer_artifacts_dir
    else:
        print("Test dataset not available. Skipping test set evaluation.")

    print("\n--- Script Finished ---")


if __name__ == "__main__":
    # Before running, ensure your callbacks.py has been updated:
    # - PlottingCallback.__init__ should accept 'plots_save_dir'
    # - OnnxExportCallback.__init__ should accept 'onnx_model_full_save_path'
    # If not, the try-except blocks for callback initialization will print warnings.
    main()
