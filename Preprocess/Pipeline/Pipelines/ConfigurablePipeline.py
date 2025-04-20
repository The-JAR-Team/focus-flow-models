import os
import json
import importlib
import time
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
# --- Use Threading ---
from concurrent.futures import ThreadPoolExecutor, as_completed
# Removed multiprocessing import

# --- Project Imports ---
try:
    from Preprocess.Pipeline import config as project_config
except ImportError:
    project_config = None

from Preprocess.Pipeline.Encapsulation.CachedTensorDataset import CachedTensorDataset
from Preprocess.Pipeline.OrchestrationPipeline import OrchestrationPipeline
from Preprocess.Pipeline.Stages.SourceStage import SourceStage
# Stage classes will be imported dynamically


# ----- Worker Function (simplified for threading) -----

def _process_single_row_thread_worker(row_dict: Dict, pipeline_instance: 'ConfigurablePipeline') -> Dict:
    """
    Worker function executed by each thread. Processes a single row.
    Accesses the main pipeline instance directly (safe with threads).
    """
    status = 'error'; error_msg = None
    clip_folder = str(row_dict.get('clip_folder', 'UnknownClip'))
    dataset_type = row_dict['dataset_type'] # Injected by process_dataset

    try:
        # Calculate Cache Path using the pipeline instance's method
        cache_file = pipeline_instance._get_expected_cache_path(row_dict, dataset_type)

        if os.path.exists(cache_file):
            return {'status': 'skipped', 'clip': clip_folder}

        # Run Pipeline using the shared inner_pipeline instance
        _ = pipeline_instance.inner_pipeline.run(data=row_dict, verbose=False) # Keep threads quiet
        status = 'processed'
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        status = 'error'

    return {'status': status, 'clip': clip_folder, 'error': error_msg}


# ----- Updated ConfigurablePipeline Class -----

class ConfigurablePipeline:
    """ Runs a preprocessing pipeline configured via JSON, using threading for row processing. """

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

        # Build Inner Pipeline instance directly, will be shared by threads
        # print("Building inner pipeline...") # Less verbose
        self.inner_pipeline = self._build_inner_pipeline(self.config.get('stages', []))
        # print(f"Inner pipeline stages: {[s.__class__.__name__ for s in self.inner_pipeline.stages]}") # Less verbose

        self.data_loader_params = self.config.get('data_loader_params', {})
        self.dataset_types_to_process = self.config.get('dataset_types_to_process', ['Train', 'Validation', 'Test'])

        print(f"Initialized Pipeline: '{self.config_name}' / v{self.pipeline_version} for '{self.dataset_name}'")

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

    # --- Renamed method to avoid confusion ---
    def _process_rows_for_dataset(self, df: pd.DataFrame, dataset_type: str, max_workers: Optional[int], overall_progress_bar: tqdm) -> Tuple[int, int, int]:
        """ Process rows from a given DataFrame using ThreadPoolExecutor, updating an overall progress bar. """
        # This method now processes a specific dataframe and updates the external bar.

        total_rows_in_df = len(df)
        num_workers = max(1, max_workers or (os.cpu_count() * 2 if os.cpu_count() else 4))
        num_workers = min(num_workers, total_rows_in_df)

        processed, skipped, errors = 0, 0, 0

        futures = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict['dataset_type'] = dataset_type
                future = executor.submit(_process_single_row_thread_worker, row_dict, self)
                futures.append(future)

            # Use as_completed without creating a new tqdm bar
            for future in as_completed(futures):
                try:
                    result = future.result()
                    status = result.get('status', 'error')
                    if status == 'processed': processed += 1
                    elif status == 'skipped': skipped += 1
                    elif status == 'error':
                        errors += 1; clip = result.get('clip', 'N/A'); error_msg = result.get('error', 'Unknown error')
                        overall_progress_bar.write(f"\n! Thread Error (clip: {clip}) ! Error: {error_msg[:500]}...")
                except Exception as e:
                    errors += 1; overall_progress_bar.write(f"\n! Critical error fetching result from thread ! Error: {e}")

                # --- Update the OVERALL progress bar ---
                overall_progress_bar.update(1)
                # Update postfix on the overall bar
                if (processed + skipped + errors) % 20 == 0 or errors > 0: # Update less frequently
                     overall_progress_bar.set_postfix(Processed=processed, Skipped=skipped, Errors=errors, refresh=False)

        # Return counts for this dataset type
        return processed, skipped, errors


    def create_dataloader(self, dataset_type: str) -> Optional[DataLoader]:
        """ Create DataLoader from cached results for a dataset type. """
        # print(f"\n--- Creating DataLoader for: {dataset_type} ---") # Less verbose
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
            # print(f"Found {len(dataset)} samples in {cache_results_dir} (recursive search).") # Less verbose
            if len(dataset) == 0: print(f"Warning: No samples found for DataLoader in {cache_results_dir}."); return None
            should_shuffle = (dataset_type.lower() == 'train')
            return DataLoader(dataset, batch_size=batch_size, shuffle=should_shuffle,
                              num_workers=num_workers, pin_memory=pin_memory)
        except Exception as e:
            print(f"Error creating DataLoader for {dataset_type} from {cache_results_dir}: {e}")
            return None

    # --- Modified run method ---
    def run(self, max_workers: Optional[int] = None) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        """ Run pipeline for configured datasets with a single overall progress bar. """
        print(f"\n=== Starting Pipeline Run: '{self.config_name}' / v{self.pipeline_version} ===")
        run_start_time = time.time()

        # 1. Ensure split files are created/cached
        print("Preparing dataset splits...")
        source_data = self.source_stage.process(verbose=False)

        # 2. Load dataframes and calculate total rows for overall progress
        dfs_to_process = []
        total_rows = 0
        print("Loading dataframes to calculate total size...")
        for ds_type in self.dataset_types_to_process:
            df = None
            try:
                if ds_type.lower() == 'train': df = source_data.get_train_data()
                elif ds_type.lower() in ['validation', 'val']: df = source_data.get_validation_data()
                elif ds_type.lower() == 'test': df = source_data.get_test_data()

                if df is not None and not df.empty:
                    dfs_to_process.append({'type': ds_type, 'df': df})
                    total_rows += len(df)
                    print(f"  Loaded {ds_type}: {len(df)} rows")
                else:
                    print(f"  Warning: No data loaded for {ds_type}.")
            except FileNotFoundError: print(f"  Warning: Split CSV not found for {ds_type}.")
            except Exception as e: print(f"  Error loading split CSV for {ds_type}: {e}.")

        if total_rows == 0:
            print("Error: No rows found to process across all specified dataset types.")
            return (None, None, None)

        # 3. Create the single overall progress bar
        print(f"\nTotal rows to process across all datasets: {total_rows}")
        overall_progress_bar = tqdm(total=total_rows, desc="Overall Progress", unit="row", leave=True)
        total_processed, total_skipped, total_errors = 0, 0, 0

        # 4. Process each dataframe sequentially, updating the overall bar using threads internally
        for data_info in dfs_to_process:
            ds_type = data_info['type']
            df = data_info['df']
            overall_progress_bar.set_description(f"Processing {ds_type}") # Update description
            # Call the processing method for this specific dataframe
            processed, skipped, errors = self._process_rows_for_dataset(
                df, ds_type, max_workers, overall_progress_bar
            )
            # Accumulate totals (optional, as postfix updates within the method)
            total_processed += processed
            total_skipped += skipped
            total_errors += errors
            # Ensure final postfix for this dataset is shown on the overall bar
            overall_progress_bar.set_postfix(Processed=total_processed, Skipped=total_skipped, Errors=total_errors, refresh=True)


        overall_progress_bar.close() # Close the main bar

        # 5. Create DataLoaders
        print("\n--- Creating DataLoaders ---")
        loaders = {ds_type.lower(): self.create_dataloader(ds_type) for ds_type in self.dataset_types_to_process}

        print(f"\n=== Pipeline Run Finished ({self.config_name} / v{self.pipeline_version}) | Total time: {(time.time() - run_start_time):.2f}s ===")
        print(f"Final Counts: Processed={total_processed}, Skipped={total_skipped}, Errors={total_errors}")
        return (loaders.get('train'), loaders.get('validation') or loaders.get('val'), loaders.get('test'))


def run_threaded_pipeline(config_path: str, max_workers: int = 1) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """ Run the pipeline with threading support. pipeline.run(max_workers=max_workers)"""
    train_loader, val_loader, test_loader = None, None, None

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at '{config_path}'")
        exit()
    else:
        print(f"\n--- Loading Config and Running Pipeline from: {config_path} ---")
        try:
            pipeline = ConfigurablePipeline(config_path=config_path)
            train_loader, val_loader, test_loader = pipeline.run(max_workers=max_workers)

            if train_loader: print(f"\nTrain DataLoader created with {len(train_loader)} batches.")
            if val_loader: print(f"Validation DataLoader created with {len(val_loader)} batches.")
            if test_loader: print(f"Test DataLoader created with {len(test_loader)} batches.")

        except Exception as e:
             print(f"\n!!! Pipeline execution failed: {e} !!!")
             import traceback; traceback.print_exc()

    return train_loader, val_loader, test_loader


# ----- Example Usage -----
if __name__ == "__main__":
    # Removed multiprocessing.freeze_support()

    config_file_path = "./configs/ENGAGENET_10fps_quality95_randdist.json" # Or your DAISEE config path
    max_workers = 4 # Adjust as needed
    run_threaded_pipeline(config_file_path, max_workers)
