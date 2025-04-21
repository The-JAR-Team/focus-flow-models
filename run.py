import os
import time
# Assuming DataLoader might be needed for type hints if used, otherwise can be removed
# from torch.utils.data import DataLoader # (Optional import if needed for hints)
from typing import Optional, Tuple, Any # For type hints

# --- Pipeline Runner Imports ---
# Direct imports - will raise ImportError if files/functions are not found
from Preprocess.Pipeline.Pipelines.CleanConfigurablePipeline import run_sequential_pipeline
from Preprocess.Pipeline.Pipelines.TowProcessesAttempt import run_two_processes_pipeline
from Preprocess.Pipeline.Pipelines.ConfigurablePipeline import run_threaded_pipeline

# --- Data Inspection Import ---
from Preprocess.Pipeline.InspectData import inspect, get_dataloader

# --- Configuration ---
# Batch size for the data inspection step
INSPECT_BATCH_SIZE = 32

# Path to the pipeline configuration JSON file (relative to this run.py script)
CONFIG_FILE_PATH = f"./Preprocess/Pipeline/Pipelines/configs/ENGAGENET_10fps_quality95_randdist.json"

# Parameters specific to the threaded execution mode
THREADED_MAX_WORKERS = 4  # Let the pipeline choose default (e.g., os.cpu_count() * 2), or set manually (e.g., 8, 16)
THREADED_CHUNK_SIZE = 128   # Number of rows processed per thread task


# === Test Functions for Each Pipeline Mode ===

def test_sequential(config_path: str):
    """Runs the pipeline sequentially."""
    print("\n" + "="*30)
    print("  RUNNING: SEQUENTIAL PIPELINE")
    print("="*30 + "\n")
    start_time = time.time()
    # Direct call - assumes run_sequential_pipeline was imported successfully
    _ = run_sequential_pipeline(config_path) # Return value (loaders) ignored here
    end_time = time.time()
    print(f"\n--- Sequential Test Duration: {end_time - start_time:.2f}s ---")

def test_two_process(config_path: str):
    """Runs the pipeline using two fixed processes."""
    print("\n" + "="*30)
    print("  RUNNING: TWO-PROCESS PIPELINE")
    print("="*30 + "\n")
    start_time = time.time()
    # Direct call - assumes run_two_processes_pipeline was imported successfully
    run_two_processes_pipeline(config_path) # This function doesn't return loaders
    end_time = time.time()
    print(f"\n--- Two-Process Test Duration: {end_time - start_time:.2f}s ---")


def test_threaded(config_path: str, workers: Optional[int], chunk_sz: int):
    """Runs the pipeline using threads with chunking."""
    print("\n" + "="*30)
    print(f"  RUNNING: THREADED PIPELINE (Workers: {workers or 'Default'}, Chunk Size: {chunk_sz})")
    print("="*30 + "\n")
    start_time = time.time()
    # Direct call - assumes run_threaded_pipeline was imported successfully
    _ = run_threaded_pipeline(config_path, max_workers=workers, chunk_size=chunk_sz) # Return value (loaders) ignored here
    end_time = time.time()
    print(f"\n--- Threaded Test Duration: {end_time - start_time:.2f}s ---")


# === Main Execution Block ===

if __name__ == "__main__":

    # --- Verify Config Path ---
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"FATAL ERROR: Configuration file not found at '{CONFIG_FILE_PATH}'")
        exit(1) # Exit if config is missing

    # --- Select ONE Pipeline Mode to Run ---
    # Uncomment the line corresponding to the test you want to perform.
    # Only one test function should be uncommented at a time for clear comparison.

    test_sequential(CONFIG_FILE_PATH)
    # test_two_process(CONFIG_FILE_PATH)
    # test_threaded(CONFIG_FILE_PATH, workers=THREADED_MAX_WORKERS, chunk_sz=THREADED_CHUNK_SIZE)

    print("\n" + "="*30)
    print("  PIPELINE EXECUTION COMPLETE")
    print("="*30 + "\n")


    # --- Inspect Results (Runs after the selected pipeline finishes) ---
    print(f"\n--- Inspecting Results in: {CONFIG_FILE_PATH} ---")
    # Direct call - assumes inspect was imported successfully
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            inspect(CONFIG_FILE_PATH, 64)
            print("\n--- Inspection Complete ---")
        except Exception as e:
            # Still keep error handling for the *execution* of inspect
            print(f"\n!!! Error during inspection: {e} !!!")
            import traceback
            traceback.print_exc()
    else:
        print(f"Warning: Base results directory for inspection not found: {CONFIG_FILE_PATH}")

    train_loader = get_dataloader(CONFIG_FILE_PATH, "train", INSPECT_BATCH_SIZE, num_workers_override=0, shuffle=True)
    val_loader = get_dataloader(CONFIG_FILE_PATH, "Validation", INSPECT_BATCH_SIZE, num_workers_override=0)
    test_loader = get_dataloader(CONFIG_FILE_PATH, "test", INSPECT_BATCH_SIZE, num_workers_override=0)
    print("\n--- run.py Finished ---")
