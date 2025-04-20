import os
import json
import importlib
import time
from typing import Dict, List, Any, Tuple, Optional
# import logging # Not used
import pandas as pd
# Lazy import DataLoader components later
# from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing

# --- Project Imports ---
try:
    from Preprocess.Pipeline import config as project_config
except ImportError:
    project_config = None

# --- Assuming these imports exist ---
try:
    # Only import base classes needed for initialization here
    from Preprocess.Pipeline.Encapsulation.CachedTensorDataset import CachedTensorDataset
    from Preprocess.Pipeline.OrchestrationPipeline import OrchestrationPipeline
    from Preprocess.Pipeline.Stages.SourceStage import SourceStage
except ImportError as e:
    print(f"Error importing core pipeline components: {e}. Please ensure they exist.")
    exit(1)


# ----- ConfigurablePipeline Class -----
class ConfigurablePipeline:
    """ Configures and runs preprocessing stages sequentially within a single process. """
    STAGE_COMMON_PARAMS = {
        "FrameExtractionStage": ["pipeline_version", "dataset_root", "cache_dir"],
        "TensorSavingStage": ["pipeline_version", "cache_dir", "config_name"],
    }

    # __init__ and _build_inner_pipeline, _get_expected_cache_path remain the same
    # as the previous two-process version.
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        if config_path:
            if not os.path.exists(config_path): raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_path, 'r') as f: self.config = json.load(f)
            self.config.setdefault('config_name', os.path.splitext(os.path.basename(config_path))[0])
        elif config_dict:
            self.config = config_dict; self.config.setdefault('config_name', 'dict_config')
        else: raise ValueError("Need config_path or config_dict")

        self.config_path = config_path # Store path if needed

        self.config_name = self.config['config_name']
        self.pipeline_version = self.config.get('pipeline_version', 'unversioned')
        self.dataset_name = self.config['dataset_name']
        self.metadata_path = self.config['metadata_csv_path']

        self.cache_root = self.config.get('cache_dir') or getattr(project_config, 'CACHE_DIR', None)
        if not self.cache_root: raise ValueError("cache_dir missing")
        os.makedirs(self.cache_root, exist_ok=True)

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
        self.inner_pipeline = self._build_inner_pipeline(self.config.get('stages', []))
        self.data_loader_params = self.config.get('data_loader_params', {})
        self.dataset_types_to_process = self.config.get('dataset_types_to_process', ['Train', 'Validation', 'Test'])

    def _build_inner_pipeline(self, stage_configs: List[Dict]) -> OrchestrationPipeline:
        stages = []
        for stage_config in stage_configs:
            stage_name = stage_config.get('name')
            if not stage_name: continue
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
                 print(f"\n!!! Error loading/instantiating stage '{stage_name}' from {module_path} !!!")
                 print(f"Params: {params}\nError: {e}")
                 raise e
        if not stages: print("Warning: No stages defined for inner pipeline.")
        try:
             return OrchestrationPipeline(stages=stages)
        except NameError:
             print("Error: OrchestrationPipeline class not found. Check imports.")
             raise

    def _get_expected_cache_path(self, row_data: dict, dataset_type: str) -> str:
        clip_folder = str(row_data.get('clip_folder', 'UnknownClip'))
        person_id = str(row_data.get('person', 'UnknownPerson'))
        subfolder = person_id
        filename = f"{clip_folder}_{self.pipeline_version}.pt"
        return os.path.join(self.cache_root,"PipelineResult", self.config_name, self.pipeline_version, dataset_type, subfolder, filename)

    # --- process_dataset optimized with itertuples ---
    def process_dataset(self, dataset_type: str, tqdm_position: int = 0) -> None:
        """ Process rows SEQUENTIALLY for a dataset type, optimized with itertuples. """
        print(f"\n--- Processing: {dataset_type} (PID: {os.getpid()}, TQDM Pos: {tqdm_position}) ---")
        try:
            source_data = self.source_stage.process(verbose=False)
        except Exception as e:
            print(f"Error running SourceStage for {dataset_type}: {e}"); return

        df = None
        try:
            if dataset_type.lower() == 'train': df = source_data.get_train_data()
            elif dataset_type.lower() in ['validation', 'val']: df = source_data.get_validation_data()
            elif dataset_type.lower() == 'test': df = source_data.get_test_data()
            else: raise ValueError(f"Invalid dataset_type: {dataset_type}")
        except AttributeError as e:
             print(f"Error: SourceStage result missing required method for {dataset_type}: {e}. Skipping.")
             return
        except Exception as e: print(f"Warning: Cannot load data for {dataset_type}: {e}. Skipping."); return

        if df is None or df.empty: print(f"Warning: No data for '{dataset_type}'. Skipping."); return

        total_rows = len(df)
        processed, skipped, errors = 0, 0, 0

        # Prepare column names for dictionary creation from tuples
        # Ensure 'Index' (or the actual index name) is first if using index=True
        if df.index.name is not None:
            column_names = [df.index.name] + list(df.columns)
        else:
            column_names = ['Index'] + list(df.columns)

        # Use tqdm with itertuples for potentially faster iteration
        progress_bar = tqdm(df.itertuples(index=True, name='Row'), # index=True includes index as first element
                            total=total_rows,
                            desc=f"Proc {dataset_type:<10}", # Pad desc
                            unit="row",
                            position=tqdm_position, # Assign specific line
                            leave=True, # Keep bar after completion
                            ncols=100, dynamic_ncols=True, # Fixed width, allow dynamic resize
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                           )

        for row_tuple in progress_bar:
            # Convert namedtuple/tuple to dictionary for pipeline stages
            # Check length just in case itertuples behaves unexpectedly (rare)
            if len(column_names) == len(row_tuple):
                row_dict = dict(zip(column_names, row_tuple))
            else:
                 # Fallback or log error if lengths don't match
                 print(f"Warning: Column/tuple length mismatch in {dataset_type}. Skipping row.")
                 continue # Skip this row

            row_dict['dataset_type'] = dataset_type # Inject type

            try:
                # Check cache (relatively fast I/O)
                cache_file = self._get_expected_cache_path(row_dict, dataset_type)
                if os.path.exists(cache_file):
                    skipped += 1
                    if skipped % 100 == 0: # Update less often for skips
                         progress_bar.set_postfix(Processed=processed, Skipped=skipped, Errors=errors, refresh=False)
                    continue
            except Exception as e:
                 errors += 1; clip = row_dict.get('clip_folder', 'N/A')
                 tqdm.write(f"! Cache check error row (clip: {clip}) in {dataset_type} ! Error: {e}")
                 continue # Skip if cache check fails

            # --- Actual Processing via Inner Pipeline ---
            # This is likely the most time-consuming part per row
            try:
                # Pass a copy if stages might modify the dict, otherwise pass directly
                _ = self.inner_pipeline.run(data=row_dict.copy(), verbose=False)
                processed += 1
            except Exception as e:
                errors += 1; clip = row_dict.get('clip_folder', 'N/A')
                # Use tqdm.write to print errors without messing up the bar
                tqdm.write(f"\n! Error processing row (clip: {clip}) in {dataset_type} ! Error: {e}")
                # Optional: Add traceback here for debugging process errors
                # import traceback; tqdm.write(traceback.format_exc())

            # Update postfix stats
            # Update reasonably frequently to see progress, refresh=False often smoother
            if (processed + errors) % 20 == 0 or errors > 0:
                 progress_bar.set_postfix(Processed=processed, Skipped=skipped, Errors=errors, refresh=False)

        progress_bar.close() # Ensure bar finishes cleanly
        print(f"--- Finished {dataset_type}: Processed={processed}, Skipped={skipped}, Errors={errors} ---")


    def create_dataloader(self, dataset_type: str) -> Optional[Any]: # Use Any for DataLoader
        """ Create DataLoader from cached results for a dataset type. """
        # Lazy import required components
        try:
            import torch
            from torch.utils.data import DataLoader
            from Preprocess.Pipeline.Encapsulation.CachedTensorDataset import CachedTensorDataset
        except ImportError as e:
            print(f"Error: PyTorch/DataLoader components required but not found ({e})."); return None

        # print(f"\n--- Creating DataLoader: {dataset_type} ---") # Less verbose
        dl_params = self.data_loader_params
        batch_size = dl_params.get('batch_size', 32)
        dataloader_num_workers = dl_params.get('num_workers', 0) # DataLoader workers
        pin_memory = dl_params.get('pin_memory', torch.cuda.is_available()) and dataloader_num_workers > 0

        cache_results_dir = os.path.join(self.cache_root,"PipelineResult", self.config_name, self.pipeline_version, dataset_type)

        if not os.path.isdir(cache_results_dir):
             print(f"Warning: DataLoader cache dir not found: {cache_results_dir}.")
             return None
        try:
            dataset = CachedTensorDataset(cache_results_dir)
            if len(dataset) == 0: print(f"Warning: No samples found for DataLoader in {cache_results_dir}."); return None
            should_shuffle = (dataset_type.lower() == 'train')
            return DataLoader(dataset, batch_size=batch_size, shuffle=should_shuffle,
                              num_workers=dataloader_num_workers, pin_memory=pin_memory,
                              persistent_workers=True if dataloader_num_workers > 0 else False)
        except NameError:
             print("Error: CachedTensorDataset class not found. Check imports.")
             return None
        except Exception as e:
            print(f"Error creating DataLoader for {dataset_type} from {cache_results_dir}: {e}")
            return None


# ----- Target Function for Multiprocessing -----
# (No changes needed here, still calls the optimized process_dataset)
def run_pipeline_splits(config_path: str, dataset_types: List[str], tqdm_position: int):
    """ Target function for a worker process using specific tqdm position. """
    process_id = os.getpid()
    try:
        pipeline = ConfigurablePipeline(config_path=config_path)
        for ds_type in dataset_types:
            pipeline.process_dataset(ds_type, tqdm_position=tqdm_position)
        print(f"Process {process_id} finished: {dataset_types}")
    except Exception as e:
        print(f"!!! Error in process {process_id} for {dataset_types}: {e} !!!")
        import traceback; traceback.print_exc()


# ----- Function to run exactly two processes -----
# (No changes needed here, still manages the two processes)
def run_two_processes_pipeline(config_path: str):
    """ Sets up and runs the preprocessing pipeline using exactly two processes. """
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: '{config_path}'"); return
    print(f"\n--- Loading Config: {config_path} ---")
    try:
        with open(config_path, 'r') as f: json.load(f)
    except Exception as e: print(f"Error reading/parsing config: {e}"); return

    process1_datasets = ['Train']
    process2_datasets = ['Validation', 'Test']
    print(f"Assigning P1 (pos 0): {process1_datasets}, P2 (pos 1): {process2_datasets}")

    process1 = multiprocessing.Process(target=run_pipeline_splits, args=(config_path, process1_datasets, 0), name="Process-Train")
    process2 = multiprocessing.Process(target=run_pipeline_splits, args=(config_path, process2_datasets, 1), name="Process-ValTest")

    processes_to_run = [p for p in [process1, process2] if p is not None] # Should always be 2 unless lists are empty
    if not processes_to_run: print("Warning: No processes assigned."); return

    print("\n--- Starting Parallel Processing (2 Processes) ---")
    run_start_time = time.time()
    for p in processes_to_run: p.start()
    for p in processes_to_run: p.join()
    run_end_time = time.time()
    print(f"\n--- All Processing Finished | Total time: {(run_end_time - run_start_time):.2f}s ---")


# ----- Example Usage -----
if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Optional: Set start method if needed
    # try: multiprocessing.set_start_method('spawn', force=True)
    # except (RuntimeError, ValueError): pass

    config_file_path = "./configs/ENGAGENET_10fps_quality95_randdist.json" # Adjust path as needed

    # --- Run parallel processing ---
    run_two_processes_pipeline(config_file_path)

    # --- Create DataLoaders post-processing ---
    print("\n--- Creating DataLoaders (Post-Processing) ---")
    try:
        # Re-initialize pipeline instance in the main process
        final_pipeline = ConfigurablePipeline(config_path=config_file_path)
        train_loader = final_pipeline.create_dataloader('Train')
        val_loader = final_pipeline.create_dataloader('Validation') or final_pipeline.create_dataloader('Val')
        test_loader = final_pipeline.create_dataloader('Test')

        # Report results concisely
        if train_loader: print(f"Train DataLoader created.")
        if val_loader: print(f"Validation DataLoader created.")
        if test_loader: print(f"Test DataLoader created.")

    except FileNotFoundError:
         print(f"\n!!! Error: Config file '{config_file_path}' not found for DataLoader creation. !!!")
    except Exception as e:
        print(f"\n!!! Error creating DataLoaders: {e} !!!")
        # import traceback; traceback.print_exc() # Uncomment for detailed debug

    print("\n--- Script Finished ---")
