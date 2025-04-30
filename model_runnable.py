# model_runnable.py (or your main script)
import torch
import torch.nn as nn
import torch.optim as optim
import os
import traceback
import time

# --- Import Configuration ---
# Import specific config variables AND the instantiated layer list
import Model.model_config as config

# --- Import Model Definition ---
# Import the *new* configurable model class
from Model.engagement_regression_model import ConfigurableEngagementModel # Adjust filename if needed

# --- Import Helper Functions (Keep relevant ones) ---
from Model.utils import get_targets, map_score_to_class_idx
from Model.training import train_model
from Model.evaluation import evaluate_model, plot_training_history
from Model.onnx_export import export_to_onnx
from Model.predict import predict_engagement

# --- Import Data Loader (Keep as is) ---
try:
    from Preprocess.Pipeline.InspectData import get_dataloader
except ImportError:
    print("ERROR: Could not import get_dataloader...")
    exit()


# ================================================
# === Main Execution ===
# ================================================
if __name__ == "__main__":
    overall_start_time = time.time()
    print("--- Starting Engagement Prediction Script (Configurable Model) ---")

    # --- Print Configuration ---
    config.print_config() # Use the updated print function from config

    # --- Create Save Directory ---
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # --- Load Data (Keep as is, validate INPUT_DIM) ---
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = None, None, None
    SEQ_LEN, NUM_LANDMARKS, NUM_COORDS = None, None, None # For ONNX
    try:
        train_loader = get_dataloader(config.CONFIG_PATH, 'Train', batch_size_override=config.BATCH_SIZE)
        val_loader = get_dataloader(config.CONFIG_PATH, 'Validation', batch_size_override=config.BATCH_SIZE)
        test_loader = get_dataloader(config.CONFIG_PATH, 'Test', batch_size_override=config.BATCH_SIZE)
        if not train_loader or not val_loader or not test_loader:
            raise ValueError("One or more dataloaders failed to initialize.")

        # Infer input shape from data and validate against config.INPUT_DIM
        print("Inferring input shape and validating...")
        sample_inputs, _ = next(iter(train_loader))
        if isinstance(sample_inputs, torch.Tensor) and sample_inputs.ndim == 4:
             SEQ_LEN = sample_inputs.shape[1]
             NUM_LANDMARKS = sample_inputs.shape[2]
             NUM_COORDS = sample_inputs.shape[3]
             ACTUAL_INPUT_DIM = NUM_LANDMARKS * NUM_COORDS
             print(f"  Inferred from data: Seq Len={SEQ_LEN}, Landmarks={NUM_LANDMARKS}, Coords={NUM_COORDS} (Input Dim={ACTUAL_INPUT_DIM})")
             if ACTUAL_INPUT_DIM != config.INPUT_DIM:
                 print(f"\n!!! FATAL ERROR: Inferred input dim ({ACTUAL_INPUT_DIM}) differs from configured INPUT_DIM ({config.INPUT_DIM}). !!!")
                 print(f"    Ensure config.INPUT_DIM in model_config.py matches data (landmarks * coords).")
                 exit()
             else:
                  print("  Inferred input dimension matches configured INPUT_DIM.")
        else:
             raise ValueError(f"Could not infer input shape. Expected 4D tensor, got {sample_inputs.ndim}D tensor.")

        # Placeholder logic for ONNX shape if inference failed (less likely now due to exit)
        # ... (keep your placeholder logic if needed) ...

    except StopIteration:
         print("\n!!! ERROR: Training dataloader is empty. Cannot validate shape or train. !!!")
         exit()
    except Exception as e:
        print(f"\n!!! ERROR during DataLoader creation or shape validation: {e} !!!")
        traceback.print_exc()
        exit()
    print("Datasets loaded successfully.")


    # --- Initialize Model (NOW USING INSTANTIATED LIST) ---
    model = None
    print("\nInitializing configurable model with layers from config...")
    try:
        # Instantiate the model, passing the pre-instantiated list from config
        model_instance = ConfigurableEngagementModel(
            input_dim=config.INPUT_DIM,
            model_layers=config.MODEL_LAYERS # Pass the list directly
        ).to(config.DEVICE)

        # --- Load Saved State Logic (Keep as is) ---
        # IMPORTANT: Saved state MUST match the exact layer structure defined in MODEL_LAYERS
        if config.LOAD_SAVED_STATE:
            if os.path.exists(config.MODEL_SAVE_PATH_PTH):
                print(f"Attempting to load saved state from: {config.MODEL_SAVE_PATH_PTH}")
                try:
                    # Load state dict - keys must match the structure in MODEL_LAYERS
                    model_instance.load_state_dict(torch.load(config.MODEL_SAVE_PATH_PTH, map_location=config.DEVICE))
                    model = model_instance
                    print("Model state loaded successfully.")
                except Exception as e:
                    print(f"\nWarning: Failed to load state dict from {config.MODEL_SAVE_PATH_PTH}: {e}")
                    print("   Ensure the saved model's architecture EXACTLY matches the current MODEL_LAYERS in config.")
                    print("   Proceeding with newly initialized model.")
                    # If loading fails, model remains None, handled below
            else:
                print(f"Warning: Saved state file not found at {config.MODEL_SAVE_PATH_PTH}.")
                print("Proceeding with newly initialized model.")

        if model is None:
            print("Using newly initialized model based on config layers.")
            model = model_instance # Use the fresh instance

        # --- Initialize Optimizer and Loss (Keep as is) ---
        criterion = nn.MSELoss() # Still MSE for regression
        # Optimizer works on parameters of the registered layers in the model
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

        print("\nConfigurable Model Summary:")
        print(model) # Print the model (will show ModuleList)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {num_params:,}")

    except Exception as e:
        print(f"\n!!! ERROR during configurable model initialization or loading: {e} !!!")
        traceback.print_exc()
        exit()

    # --- Train (Pass the configurable model instance) ---
    # train_model helper function doesn't need changes if it uses model generically
    trained_model = None
    history = None
    skip_training = False # Your logic here
    if skip_training:
        print("\nSkipping training as requested.")
        trained_model = model
    else:
        print("\nStarting model training process...")
        try:
            trained_model, history = train_model(
                model=model, # Pass the configurable model
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=config.NUM_EPOCHS,
                device=config.DEVICE,
                save_path_pth=config.MODEL_SAVE_PATH_PTH,
                save_best_pth=config.SAVE_BEST_MODEL_PTH,
                label_to_idx_map=config.LABEL_TO_IDX_MAP,
                idx_to_score_map=config.IDX_TO_SCORE_MAP
            )
        except KeyboardInterrupt:
             print("\n--- Training interrupted by user ---")
             trained_model = model # Use model at interruption
             history = None
             print("Proceeding with model state at interruption.")
        except Exception as e:
             print(f"\n!!! ERROR during training: {e} !!!")
             traceback.print_exc()
             # Attempt to load best saved model if available
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
                 trained_model = None
             history = None


    # --- Plot Training History (Unchanged) ---
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


    # --- Evaluate (Pass the configurable model instance) ---
    # evaluate_model helper function doesn't need changes
    if trained_model and test_loader:
        print("\nStarting evaluation process...")
        try:
            snp_idx_to_use = 4 # Or get dynamically
            test_loss, multi_class_acc, binary_acc = evaluate_model(
                model=trained_model, # Pass the configurable model
                test_loader=test_loader,
                criterion=criterion,
                device=config.DEVICE,
                label_to_idx_map=config.LABEL_TO_IDX_MAP,
                idx_to_score_map=config.IDX_TO_SCORE_MAP,
                idx_to_name_map=config.IDX_TO_NAME_MAP,
                confusion_matrix_path=config.CONFUSION_MATRIX_PATH,
                snp_index=snp_idx_to_use
            )
            print(f"\nEvaluation Summary: Test Loss={test_loss:.4f}, Multi-Class Acc={multi_class_acc:.4f}, Binary Acc={binary_acc:.4f}")
        except Exception as e:
            print(f"\n!!! ERROR during evaluation: {e} !!!")
            traceback.print_exc()
    elif not trained_model:
        print("\nSkipping evaluation: No valid trained model available.")
    else: # trained_model exists but test_loader doesn't
         print("\nSkipping evaluation: Test loader not available.")


    # --- Export to ONNX (Pass the configurable model instance) ---
    # onnx_export helper function doesn't need changes
    if trained_model and config.SAVE_FINAL_MODEL_ONNX:
        if SEQ_LEN is not None and NUM_LANDMARKS is not None and NUM_COORDS is not None:
            print("\nStarting ONNX export process...")
            try:
                # Create dummy input based on inferred shape
                dummy_input = torch.randn(1, SEQ_LEN, NUM_LANDMARKS, NUM_COORDS, device='cpu')
                export_to_onnx(
                    model=trained_model, # Pass the configurable model
                    dummy_input=dummy_input,
                    save_path_onnx=config.MODEL_SAVE_PATH_ONNX,
                    device=config.DEVICE,
                    opset_version=config.ONNX_OPSET_VERSION
                )
            except Exception as e:
                print(f"\n!!! ERROR during ONNX export preparation or execution: {e} !!!")
                traceback.print_exc()
        else:
             print("\n!!! ERROR during ONNX export: Could not determine input shape for dummy input. Export skipped. !!!")
    elif not trained_model:
         print("\nSkipping ONNX export: No valid trained model.")
    elif not config.SAVE_FINAL_MODEL_ONNX:
        print("\nSkipping ONNX export: Saving disabled in configuration.")


    # --- Prediction Example (Pass the configurable model instance) ---
    # predict_engagement helper function doesn't need changes
    run_prediction_example = False # Set to True to run
    if run_prediction_example and trained_model and test_loader:
         print("\nRunning prediction example on test set...")
         try:
             predict_engagement(
                 model=trained_model, # Pass the configurable model
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

