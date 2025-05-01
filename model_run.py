import torch
import torch.nn as nn
import torch.optim as optim
import os
import traceback
import time

# --- Import Configuration ---
import Model.model_config as config

# --- Import Functions ---
from Model.utils import get_targets # Only need get_targets here if used directly, otherwise handled within modules
from Model.training import train_model
from Model.evaluation import evaluate_model, plot_training_history
from Model.onnx_export import export_to_onnx
from Model.predict import predict_engagement

# --- Import Data Loader ---
# Assuming get_dataloader is defined elsewhere and handles data loading/preprocessing
try:
    from Preprocess.Pipeline.InspectData import get_dataloader
except ImportError:
    print("ERROR: Could not import get_dataloader from Preprocess.Pipeline.InspectData.")
    print("       Please ensure the necessary preprocessing modules are accessible.")
    exit()


# ================================================
# === Main Execution ===
# ================================================
if __name__ == "__main__":
    overall_start_time = time.time()
    print("--- Starting Engagement Prediction Script ---")

    # --- Print Configuration ---
    config.print_config() # Print settings from config.py

    # --- Create Save Directory ---
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # --- Load Data ---
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = None, None, None
    # Define variables for ONNX export shape outside the try block
    SEQ_LEN, NUM_LANDMARKS, NUM_COORDS = None, None, None
    try:
        # Load data using the provided function and config settings
        train_loader = get_dataloader(config.CONFIG_PATH, 'Train', batch_size_override=config.BATCH_SIZE)
        val_loader = get_dataloader(config.CONFIG_PATH, 'Validation', batch_size_override=config.BATCH_SIZE)
        test_loader = get_dataloader(config.CONFIG_PATH, 'Test', batch_size_override=config.BATCH_SIZE)
        if not train_loader or not val_loader or not test_loader:
            raise ValueError("One or more dataloaders failed to initialize.")

        # Infer input shape from a sample batch for ONNX export
        print("Inferring input shape for ONNX export...")
        try:
            sample_inputs, _ = next(iter(train_loader))
            if isinstance(sample_inputs, torch.Tensor) and sample_inputs.ndim == 4:
                 SEQ_LEN = sample_inputs.shape[1]
                 NUM_LANDMARKS = sample_inputs.shape[2]
                 NUM_COORDS = sample_inputs.shape[3]
                 ACTUAL_INPUT_DIM = NUM_LANDMARKS * NUM_COORDS
                 print(f"  Inferred from data: Seq Len={SEQ_LEN}, Landmarks={NUM_LANDMARKS}, Coords={NUM_COORDS} (Input Dim={ACTUAL_INPUT_DIM})")
                 # Optional: Check consistency with configured INPUT_DIM
                 if ACTUAL_INPUT_DIM != config.INPUT_DIM:
                     print(f"  Warning: Inferred input dim ({ACTUAL_INPUT_DIM}) differs from configured INPUT_DIM ({config.INPUT_DIM}).")
                     print(f"           Using configured INPUT_DIM ({config.INPUT_DIM}) for model initialization.")
                     print(f"           Using inferred shape ({SEQ_LEN}, {NUM_LANDMARKS}, {NUM_COORDS}) for ONNX dummy input.")
            else:
                # Handle cases where the data might not be as expected
                raise ValueError(f"Could not infer input shape. Expected 4D tensor, got {sample_inputs.ndim}D tensor of type {type(sample_inputs)}.")
        except StopIteration:
             raise ValueError("Training dataloader is empty, cannot infer shape.")
        except Exception as e_infer:
             print(f"  Warning: Could not infer input shape from data ({e_infer}).")
             # Attempt to use placeholder values ONLY if inference failed and shape is needed
             if config.SAVE_FINAL_MODEL_ONNX and (SEQ_LEN is None or NUM_LANDMARKS is None or NUM_COORDS is None):
                 print("  Attempting to use placeholder values for ONNX export shape.")
                 SEQ_LEN = 30 # Example placeholder sequence length
                 # Try to derive landmarks/coords from INPUT_DIM assuming 3 coords
                 if config.INPUT_DIM % 3 == 0:
                     NUM_LANDMARKS = config.INPUT_DIM // 3
                     NUM_COORDS = 3
                     print(f"  Using placeholders: Seq Len={SEQ_LEN}, Landmarks={NUM_LANDMARKS}, Coords={NUM_COORDS}")
                 else:
                      # Cannot determine shape, ONNX export will likely fail later
                      print(f"  ERROR: Cannot determine placeholder shape for ONNX export from INPUT_DIM={config.INPUT_DIM}.")
                      # Set flags to prevent ONNX export attempt later if shape is unknown
                      SEQ_LEN, NUM_LANDMARKS, NUM_COORDS = None, None, None


    except Exception as e:
        print(f"\n!!! ERROR during DataLoader creation or shape inference: {e} !!!")
        traceback.print_exc()
        exit()
    print("Datasets loaded successfully.")

    # --- Initialize Model ---
    model = None
    print("\nInitializing model...")
    try:
        # Instantiate model definition using configured INPUT_DIM
        model_instance = config.get_model()

        # Attempt to load saved state if requested
        if config.LOAD_SAVED_STATE:
            if os.path.exists(config.MODEL_SAVE_PATH_PTH):
                print(f"Attempting to load saved state from: {config.MODEL_SAVE_PATH_PTH}")
                try:
                    # Load state dict into the instantiated model
                    model_instance.load_state_dict(torch.load(config.MODEL_SAVE_PATH_PTH, map_location=config.DEVICE))
                    model = model_instance # Assign loaded model
                    print("Model state loaded successfully.")
                except Exception as e:
                    print(f"Warning: Failed to load state dict from {config.MODEL_SAVE_PATH_PTH}: {e}")
                    print("Proceeding with newly initialized model.")
            else:
                print(f"Warning: Saved state file not found at {config.MODEL_SAVE_PATH_PTH}.")
                print("Proceeding with newly initialized model.")

        # If model wasn't loaded (either disabled or failed), use the fresh instance
        if model is None:
            print("Using newly initialized model.")
            model = model_instance

        model.to(config.DEVICE)

        # --- Initialize Optimizer and Loss ---
        # Using Mean Squared Error for regression
        criterion = nn.MSELoss()
        # Using AdamW optimizer
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

        print("\nModel Summary:")
        print(model)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {num_params:,}")

    except Exception as e:
        print(f"\n!!! ERROR during model initialization or loading: {e} !!!")
        traceback.print_exc()
        exit()

    # --- Train ---
    trained_model = None
    history = None
    # Decide whether to train based on LOAD_SAVED_STATE or other flags if needed
    # Currently, it trains even if a state is loaded. Add logic here to skip if desired.
    skip_training = False # Example flag, set based on LOAD_SAVED_STATE if needed
    if skip_training:
         print("\nSkipping training as requested or due to loaded state.")
         trained_model = model # Use the loaded/initialized model
    else:
        print("\nStarting model training process...")
        try:
            # Call the training function from training.py
            trained_model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=config.NUM_EPOCHS,
                device=config.DEVICE,
                save_path_pth=config.MODEL_SAVE_PATH_PTH,
                save_best_pth=config.SAVE_BEST_MODEL_PTH,
                label_to_idx_map=config.LABEL_TO_IDX_MAP, # Pass necessary maps
                idx_to_score_map=config.IDX_TO_SCORE_MAP
            )
        except KeyboardInterrupt:
             print("\n--- Training interrupted by user ---")
             # Use the model state at interruption (best saved might be loaded by train_model)
             trained_model = model
             history = None # History might be incomplete
             print("Proceeding with model state at interruption.")
        except Exception as e:
            print(f"\n!!! ERROR during training: {e} !!!")
            traceback.print_exc()
            # Attempt to load the best saved model if training failed mid-way and saving was enabled
            if config.SAVE_BEST_MODEL_PTH and os.path.exists(config.MODEL_SAVE_PATH_PTH):
                print("Attempting to load best saved model due to training error...")
                try:
                    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH_PTH, map_location=config.DEVICE))
                    trained_model = model
                    print("Successfully loaded best saved model.")
                except Exception as le:
                     print(f"Could not load best saved model: {le}. No trained model available.")
                     trained_model = None
            else:
                trained_model = None # No trained model available
            history = None # History is likely invalid


    # --- Plot Training History ---
    if history:
        print("\nPlotting training history...")
        try:
            plot_training_history(
                history=history,
                loss_curve_path=config.LOSS_CURVE_PATH,
                acc_curve_path=config.ACC_CURVE_PATH
            )
        except Exception as e:
             print(f"Error plotting history: {e}")
    else:
        print("\nSkipping history plotting: No history data available.")

    # --- Evaluate ---
    if trained_model and test_loader:
        print("\nStarting evaluation process...")
        try:
            test_loss, multi_class_acc, binary_acc = evaluate_model(
                model=trained_model,
                test_loader=test_loader,
                criterion=criterion,  # Use the same criterion for loss calculation
                device=config.DEVICE,
                label_to_idx_map=config.LABEL_TO_IDX_MAP,
                idx_to_score_map=config.IDX_TO_SCORE_MAP,
                idx_to_name_map=config.IDX_TO_NAME_MAP,
                confusion_matrix_path=config.CONFUSION_MATRIX_PATH,
            )
            # You can now use test_loss, multi_class_acc, binary_acc if needed later in the script
            print(
                f"\nEvaluation Summary: Test Loss={test_loss:.4f}, Multi-Class Acc={multi_class_acc:.4f}, Binary Acc={binary_acc:.4f}")

        except Exception as e:
            print(f"\n!!! ERROR during evaluation: {e} !!!")
            traceback.print_exc()
    elif not trained_model:
        print("\nSkipping evaluation: No valid trained model available.")
    else: # trained_model exists but test_loader doesn't
         print("\nSkipping evaluation: Test loader not available.")


    # --- Export to ONNX ---
    if trained_model and config.SAVE_FINAL_MODEL_ONNX:
        # Check if shape variables were successfully determined for dummy input
        if SEQ_LEN is not None and NUM_LANDMARKS is not None and NUM_COORDS is not None:
            print("\nStarting ONNX export process...")
            try:
                # Create the dummy input tensor on CPU first
                dummy_input = torch.randn(1, SEQ_LEN, NUM_LANDMARKS, NUM_COORDS, device='cpu')
                # Call the export function from onnx_export.py
                export_to_onnx(
                    model=trained_model,
                    dummy_input=dummy_input,
                    save_path_onnx=config.MODEL_SAVE_PATH_ONNX,
                    device=config.DEVICE, # Export will move model/input to this device
                    opset_version=config.ONNX_OPSET_VERSION
                )
            except Exception as e:
                # Catch errors during dummy input creation or export call
                print(f"\n!!! ERROR during ONNX export preparation or execution: {e} !!!")
                traceback.print_exc()
        else:
             # This case occurs if shape inference failed and placeholders couldn't be determined
             print("\n!!! ERROR during ONNX export: Could not determine input shape for dummy input. Export skipped. !!!")
             print("   Check dataloader and shape inference steps in the loading section.")

    elif not trained_model:
         print("\nSkipping ONNX export: No valid trained model.")
    elif not config.SAVE_FINAL_MODEL_ONNX:
        print("\nSkipping ONNX export: Saving disabled in configuration.")

    run_prediction_example = False
    if run_prediction_example and trained_model and test_loader:
         print("\nRunning prediction example on test set...")
         try:
             # Call the prediction function from predict.py
             predict_engagement(
                 model=trained_model,
                 data_loader=test_loader,
                 device=config.DEVICE,
                 idx_to_name_map=config.IDX_TO_NAME_MAP
             )
         except Exception as e:
             print(f"\n!!! ERROR during prediction example: {e} !!!")
             traceback.print_exc()
    elif run_prediction_example:
        print("\nSkipping prediction example: Model or test loader not available.")


    overall_end_time = time.time()
    print("\n--- Script Finished ---")
    print(f"Total script execution time: {(overall_end_time - overall_start_time):.2f}s")
# ================================================

