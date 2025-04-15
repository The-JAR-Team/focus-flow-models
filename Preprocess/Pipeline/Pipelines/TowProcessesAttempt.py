import os
import json
import importlib
import time
from typing import Dict, List, Any, Tuple, Optional
import inspect
import logging

import pandas as pd
# import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# --- Multiprocessing for dataset-level parallelism ---
import multiprocessing
# --- Concurrent Futures Removed ---

# --- Project Imports ---
try:
    from Preprocess.Pipeline import config as project_config
except ImportError:
    project_config = None

from Preprocess.Pipeline.Encapsulation.CachedTensorDataset import CachedTensorDataset
from Preprocess.Pipeline.OrchestrationPipeline import OrchestrationPipeline
from Preprocess.Pipeline.Stages.SourceStage import SourceStage
# Stage classes will be imported dynamically


# ----- ConfigurablePipeline Class -----

class ConfigurablePipeline:
    """ Runs a preprocessing pipeline configured via JSON sequentially within each process. """

    STAGE_COMMON_PARAMS = {
        "FrameExtractionStage": ["pipeline_version", "dataset_root", "cache_dir"],
        "TensorSavingStage": ["pipeline_version", "cache_dir", "config_name"],
    }

    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """ Initialize pipeline from JSON config (path or dict). """
        if config_path:
            with open(config_path, 'r') as f: self.config = json.load(f)
            self.config.setdefault('config_name', os.path.splitext(os.path.basename(config_path))[0])
        elif config_dict:
            self.config = config_dict; self.config.setdefault('config_name', 'dict_config')
        else: raise ValueError("Need config_path or config_dict")

        self.config_name = self.config['config_name']
        self.pipeline_version = self.config.get('pipeline_version', 'unversioned')
        self.dataset_name = self.config['dataset_name']
        self.metadata_path = self.config['metadata_csv_path']

        self.cache_root = self.config.get('cache_dir') or getattr(project_config, 'CACHE_DIR', None)
        if not self.cache_root: raise ValueError("cache_dir missing")

        self.dataset_root = self.config.get('dataset_root')
        if not self.dataset_root:
            root_attr = f"{self.dataset_name.upper()}_DATASET_ROOT"
            self.dataset_root = getattr(project_config, root_attr, None)
            if not self.dataset_root: raise ValueError(f"dataset_root missing and {root_attr} not in config.py")

        source_stage_cfg = self.config.get('source_stage_config', {})
        self.source_stage = SourceStage(
            pipeline_version=self.pipeline_version, cache_dir=self.cache_root,
            metadata_csv_path=self.metadata_path, **source_stage_cfg
        )

        # print("Building inner pipeline...") # Keep less verbose
        self.inner_pipeline = self._build_inner_pipeline(self.config.get('stages', []))

        self.data_loader_params = self.config.get('data_loader_params', {})
        self.dataset_types_to_process = self.config.get('dataset_types_to_process', ['Train', 'Validation', 'Test'])

        # print(f"Initialized Pipeline Instance: '{self.config_name}' / v{self.pipeline_version} for '{self.dataset_name}'") # Keep less verbose

    def _build_inner_pipeline(self, stage_configs: List[Dict]) -> OrchestrationPipeline:
        """ Dynamically build the inner pipeline from configuration list. """
        stages = []
        for stage_config in stage_configs:
            stage_name = stage_config['name']
            params = stage_config.get('params', {})

            common_needed = self.STAGE_COMMON_PARAMS.get(stage_name, [])
            if "pipeline_version" in common_needed: params['pipeline_version'] = self.pipeline_version
            if "dataset_root" in common_needed: params['dataset_root'] = self.dataset_root
            if "cache_dir" in common_needed: params['cache_dir'] = self.cache_root
            if "config_name" in common_needed: params['config_name'] = self.config_name

            try:
                module_path = f"Preprocess.Pipeline.Stages.{stage_name}"
                module = importlib.import_module(module_path)
                stage_class = getattr(module, stage_name)
                stage_instance = stage_class(**params)
                stages.append(stage_instance)
            except Exception as e:
                 print(f"\n!!! Error loading/instantiating stage '{stage_name}' !!!")
                 print(f"Module Path: {module_path}, Params Passed: {params}")
                 raise e

        if not stages: print("Warning: No stages defined for inner pipeline.")
        return OrchestrationPipeline(stages=stages)

    def _get_expected_cache_path(self, row_data: dict, dataset_type: str) -> str:
        """ Calculate final tensor cache path. MUST match TensorSavingStage logic. """
        clip_folder = str(row_data['clip_folder'])
        person_id = str(row_data.get('person', 'UnknownPerson'))
        subfolder = person_id
        filename = f"{clip_folder}_{self.pipeline_version}.pt"
        return os.path.join(self.cache_root,"PipelineResult", self.config_name, self.pipeline_version, dataset_type, subfolder, filename)

    # --- Modified process_dataset to accept tqdm_position ---
    def process_dataset(self, dataset_type: str, tqdm_position: int = 0) -> None:
        """ Process all rows SEQUENTIALLY for a given dataset type with positioned tqdm bar. """
        # Show process ID in print statement for clarity
        print(f"\n--- Processing dataset: {dataset_type} (Process: {os.getpid()}) ---")
        source_data = self.source_stage.process(verbose=False)

        df = None
        try: # Load appropriate split dataframe
            if dataset_type.lower() == 'train': df = source_data.get_train_data()
            elif dataset_type.lower() in ['validation', 'val']: df = source_data.get_validation_data()
            elif dataset_type.lower() == 'test': df = source_data.get_test_data()
            else: raise ValueError(f"Invalid dataset_type: {dataset_type}")
        except FileNotFoundError as e: print(f"Warning: Split CSV not found for {dataset_type}: {e}. Skipping."); return
        except Exception as e: print(f"Error loading split CSV for {dataset_type}: {e}. Skipping."); return

        if df is None or df.empty: print(f"Warning: No data loaded for '{dataset_type}'. Skipping."); return

        total_rows = len(df)
        # print(f"Processing {total_rows} rows for {dataset_type} sequentially...") # Replaced by tqdm desc
        processed, skipped, errors = 0, 0, 0

        # Use tqdm for sequential progress bar with position and leave=True
        # --- Added ncols argument ---
        progress_bar = tqdm(df.iterrows(), total=total_rows,
                            desc=f"Processing {dataset_type:<10}", # Pad desc for alignment
                            unit="row",
                            position=tqdm_position, # Assign specific line
                            leave=True, # Keep bar after completion
                            ncols=100 # Try setting fixed width (adjust as needed)
                           )

        for idx, row_tuple in enumerate(progress_bar):
            _, row = row_tuple; row_dict = row.to_dict()
            row_dict['dataset_type'] = dataset_type

            cache_file = self._get_expected_cache_path(row_dict, dataset_type)
            if os.path.exists(cache_file):
                skipped += 1
                if skipped % 100 == 0: # Update less often for skips
                     progress_bar.set_postfix(Processed=processed, Skipped=skipped, Errors=errors, refresh=False)
                continue

            # --- Add Timing Start ---
            row_start_time = time.time()
            # -----------------------
            try:
                verbose_run = False # Keep inner stages quiet generally
                _ = self.inner_pipeline.run(data=row_dict, verbose=verbose_run)
                processed += 1
                # --- Add Timing End & Print ---
                row_end_time = time.time()
                processing_time = row_end_time - row_start_time
                # Print timing info occasionally using tqdm.write to not mess up bars
                # if processed % 20 == 0: # Maybe disable this timing print for now
                #     clip = row_dict.get('clip_folder', 'N/A')
                #     tqdm.write(f"  ({dataset_type}) Row {idx+1} (Clip: {clip}) processed in {processing_time:.3f}s")
                # ---------------------------
            except Exception as e:
                errors += 1; clip = row_dict.get('clip_folder', 'N/A')
                # Use tqdm.write to print errors without messing up the bar
                tqdm.write(f"\n! Error row {idx + 1} (clip: {clip}) in {dataset_type} ! Error: {e}")

            # Update postfix stats more frequently now that bars are separate
            progress_bar.set_postfix(Processed=processed, Skipped=skipped, Errors=errors, refresh=False) # refresh=False often smoother

        progress_bar.close() # Ensure bar finishes cleanly
        print(f"--- Finished {dataset_type}: Processed={processed}, Skipped={skipped}, Errors={errors} ---")


    def create_dataloader(self, dataset_type: str) -> Optional[DataLoader]:
        """ Create DataLoader from cached results for a dataset type. """
        print(f"\n--- Creating DataLoader for: {dataset_type} ---")
        dl_params = self.data_loader_params
        batch_size = dl_params.get('batch_size', 32)
        num_workers = dl_params.get('num_workers', 0)
        pin_memory = dl_params.get('pin_memory', True) if num_workers > 0 else False

        cache_results_dir = os.path.join(self.cache_root,"PipelineResult", self.config_name, self.pipeline_version, dataset_type)

        if not os.path.isdir(cache_results_dir):
             print(f"Warning: Results cache dir not found: {cache_results_dir}.")
             return None
        try:
            dataset = CachedTensorDataset(cache_results_dir)
            print(f"Found {len(dataset)} samples in {cache_results_dir} (recursive search).")
            if len(dataset) == 0: return None
            should_shuffle = (dataset_type.lower() == 'train')
            return DataLoader(dataset, batch_size=batch_size, shuffle=should_shuffle,
                              num_workers=num_workers, pin_memory=pin_memory)
        except Exception as e:
            print(f"Error creating DataLoader for {dataset_type} from {cache_results_dir}: {e}")
            return None

    # Removed the run method


# ----- Target Function for Multiprocessing -----

# --- Modified run_pipeline_splits to accept tqdm_position ---
def run_pipeline_splits(config_path: str, dataset_types: List[str], tqdm_position: int):
    """
    Target function for a worker process. Initializes a pipeline from config
    and processes the specified list of dataset types sequentially, using a specific tqdm position.
    """
    # print(f"Process {os.getpid()} starting for datasets: {dataset_types} at tqdm position {tqdm_position}") # Less verbose start
    try:
        pipeline = ConfigurablePipeline(config_path=config_path)
        for ds_type in dataset_types:
            # Pass the position to process_dataset
            pipeline.process_dataset(ds_type, tqdm_position=tqdm_position)
        print(f"Process {os.getpid()} finished datasets: {dataset_types}")
    except Exception as e:
        print(f"!!! Error in process {os.getpid()} for datasets {dataset_types} !!!")
        import traceback
        traceback.print_exc()


# ----- Example Usage -----
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Use the config file path provided by the user previously
    config_file_path = "./configs/ENGAGENET_10fps_quality95_randdist.json" # Or your DAISEE config path

    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file not found at '{config_file_path}'")
        exit()

    print(f"\n--- Loading Config from: {config_file_path} ---")

    # Define the split assignments
    process1_datasets = ['Train']
    process2_datasets = ['Validation', 'Test']

    # print(f"Assigning Process 1 (tqdm pos 0) to: {process1_datasets}") # Less verbose
    # print(f"Assigning Process 2 (tqdm pos 1) to: {process2_datasets}")

    # --- Create Process objects, passing tqdm_position ---
    process1 = multiprocessing.Process(target=run_pipeline_splits, args=(config_file_path, process1_datasets, 0)) # Position 0
    process2 = multiprocessing.Process(target=run_pipeline_splits, args=(config_file_path, process2_datasets, 1)) # Position 1

    print("\n--- Starting Parallel Processing (Dataset Level) ---")
    run_start_time = time.time()

    process1.start()
    process2.start()

    process1.join()
    # print("--- Process 1 (Train) finished ---") # Less verbose
    process2.join()
    # print("--- Process 2 (Validation, Test) finished ---")

    run_end_time = time.time()
    print(f"\n--- All Processing Finished | Total time: {(run_end_time - run_start_time):.2f}s ---")

    # --- Create DataLoaders (in main process after workers finish) ---
    print("\n--- Creating DataLoaders ---")
    try:
        final_pipeline = ConfigurablePipeline(config_path=config_file_path) # Re-init to use create_dataloader
        train_loader = final_pipeline.create_dataloader('Train')
        val_loader = final_pipeline.create_dataloader('Validation') or final_pipeline.create_dataloader('Val')
        test_loader = final_pipeline.create_dataloader('Test')

        if train_loader: print(f"Train DataLoader created with {len(train_loader)} batches.")
        if val_loader: print(f"Validation DataLoader created with {len(val_loader)} batches.")
        if test_loader: print(f"Test DataLoader created with {len(test_loader)} batches.")
    except Exception as e:
        print(f"\n!!! Error creating DataLoaders: {e} !!!")
        import traceback; traceback.print_exc()
