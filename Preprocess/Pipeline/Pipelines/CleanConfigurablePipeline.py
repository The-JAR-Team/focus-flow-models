import os
import json
import importlib
import time
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
# Removed concurrent.futures and multiprocessing imports

# tqdm is needed for the sequential progress bar
from tqdm import tqdm

# --- Project Imports ---
try:
    # Attempt to import project-specific config (e.g., for CACHE_DIR defaults)
    from Preprocess.Pipeline import config as project_config
except ImportError:
    project_config = None
    # print("Note: project_config not found.") # Less verbose

# --- Assuming these imports exist in your project structure ---
try:
    # Only import base classes needed for initialization here
    # Lazy import DataLoader, CachedTensorDataset later
    from Preprocess.Pipeline.OrchestrationPipeline import OrchestrationPipeline
    from Preprocess.Pipeline.Stages.SourceStage import SourceStage
except ImportError as e:
    print(f"Error importing core pipeline components: {e}. Please ensure they exist.")
    exit(1)


# ----- ConfigurablePipeline Class (Sequential Version) -----

class ConfigurablePipeline:
    """ Runs a preprocessing pipeline configured via JSON sequentially in a single process. """

    STAGE_COMMON_PARAMS = {
        "FrameExtractionStage": ["pipeline_version", "dataset_root", "cache_dir"],
        "TensorSavingStage": ["pipeline_version", "cache_dir", "config_name"],
    }

    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """ Initialize pipeline from JSON config (path or dict). """
        if config_path:
            if not os.path.exists(config_path): raise FileNotFoundError(f"Config file not found: {config_path}")
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
        os.makedirs(self.cache_root, exist_ok=True)

        self.dataset_root = self.config.get('dataset_root')
        if not self.dataset_root:
            root_attr = f"{self.dataset_name.upper()}_DATASET_ROOT"
            self.dataset_root = getattr(project_config, root_attr, None)
            if not self.dataset_root: raise ValueError(f"dataset_root missing and {root_attr} not in config.py")

        # Initialize components needed for processing
        source_stage_cfg = self.config.get('source_stage_config', {})
        self.source_stage = SourceStage(
            pipeline_version=self.pipeline_version, cache_dir=self.cache_root,
            metadata_csv_path=self.metadata_path, **source_stage_cfg
        )
        self.inner_pipeline = self._build_inner_pipeline(self.config.get('stages', []))
        self.data_loader_params = self.config.get('data_loader_params', {})
        self.dataset_types_to_process = self.config.get('dataset_types_to_process', ['Train', 'Validation', 'Test'])

        # print(f"Initialized Sequential Pipeline: '{self.config_name}' / v{self.pipeline_version} for '{self.dataset_name}'") # Less verbose

    def _build_inner_pipeline(self, stage_configs: List[Dict]) -> OrchestrationPipeline:
        """ Dynamically build the inner pipeline from configuration list. """
        # (Same as previous versions)
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
                 print(f"\n!!! Error loading/instantiating stage '{stage_name}' !!!")
                 print(f"Module Path: {module_path}, Params Passed: {params}")
                 raise e
        if not stages: print("Warning: No stages defined for inner pipeline.")
        return OrchestrationPipeline(stages=stages)

    def _get_expected_cache_path(self, row_data: dict, dataset_type: str) -> str:
        """ Calculate final tensor cache path. MUST match TensorSavingStage logic. """
        # (Same as previous versions)
        clip_folder = str(row_data.get('clip_folder', 'UnknownClip')) # Use .get for safety
        person_id = str(row_data.get('person', 'UnknownPerson'))
        subfolder = person_id
        filename = f"{clip_folder}_{self.pipeline_version}.pt"
        # Construct path relative to cache root
        return os.path.join(self.cache_root,"PipelineResult", self.config_name, self.pipeline_version, dataset_type, subfolder, filename)

    def create_dataloader(self, dataset_type: str) -> Optional[Any]: # Use Any for DataLoader type hint
        """ Create DataLoader from cached results for a dataset type. """
        # Lazy import required components here
        try:
            import torch # Needed for pin_memory check
            from torch.utils.data import DataLoader
            from Preprocess.Pipeline.Encapsulation.CachedTensorDataset import CachedTensorDataset
        except ImportError as e:
            print(f"Error: PyTorch/DataLoader components required but not found ({e})."); return None

        # print(f"\n--- Creating DataLoader for: {dataset_type} ---") # Less verbose
        dl_params = self.data_loader_params
        batch_size = dl_params.get('batch_size', 32)
        # DataLoader's num_workers (for loading batches), not pipeline processing workers
        dataloader_num_workers = dl_params.get('num_workers', 0)
        pin_memory = dl_params.get('pin_memory', torch.cuda.is_available()) and dataloader_num_workers > 0

        cache_results_dir = os.path.join(self.cache_root,"PipelineResult", self.config_name, self.pipeline_version, dataset_type)

        if not os.path.isdir(cache_results_dir):
             print(f"Warning: Results cache dir not found for DataLoader: {cache_results_dir}.")
             return None
        try:
            dataset = CachedTensorDataset(cache_results_dir)
            # print(f"Found {len(dataset)} samples in {cache_results_dir} (recursive search).") # Less verbose
            if len(dataset) == 0: print(f"Warning: No samples found for DataLoader in {cache_results_dir}."); return None
            should_shuffle = (dataset_type.lower() == 'train')
            return DataLoader(dataset, batch_size=batch_size, shuffle=should_shuffle,
                              num_workers=dataloader_num_workers, pin_memory=pin_memory,
                              persistent_workers=True if dataloader_num_workers > 0 else False) # Use persistent workers if using workers
        except NameError:
            print("Error: CachedTensorDataset class not found. Check imports.")
            return None
        except Exception as e:
            print(f"Error creating DataLoader for {dataset_type} from {cache_results_dir}: {e}")
            return None

    # --- New sequential run method ---
    def run_sequentially(self) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """
        Run pipeline for configured datasets SEQUENTIALLY in a single process.
        Uses one overall progress bar.
        """
        print(f"\n=== Starting Sequential Pipeline Run: '{self.config_name}' / v{self.pipeline_version} ===")
        run_start_time = time.time()

        # 1. Ensure split files are created/cached by running the source stage
        print("Preparing dataset splits...")
        try:
            source_data = self.source_stage.process(verbose=False)
        except Exception as e:
            print(f"Error during source stage processing: {e}")
            return (None, None, None) # Cannot proceed

        # 2. Load dataframes and calculate total rows for overall progress bar
        all_rows_data = []
        total_rows = 0
        print("Loading dataframes and collecting all rows...")
        for ds_type in self.dataset_types_to_process:
            df = None
            try:
                if ds_type.lower() == 'train': df = source_data.get_train_data()
                elif ds_type.lower() in ['validation', 'val']: df = source_data.get_validation_data()
                elif ds_type.lower() == 'test': df = source_data.get_test_data()
                else: print(f"Warning: Unknown dataset type '{ds_type}' in config, skipping."); continue

                if df is not None and not df.empty:
                    print(f"  Loaded {ds_type}: {len(df)} rows")
                    # Store row data along with its dataset type
                    for _, row in df.iterrows():
                        row_dict = row.to_dict()
                        row_dict['dataset_type'] = ds_type # Inject type for processing step
                        all_rows_data.append(row_dict)
                    total_rows += len(df)
                else:
                    print(f"  Info: No data loaded for {ds_type}.")
            except FileNotFoundError: print(f"  Warning: Split CSV not found for {ds_type}.")
            except AttributeError as e: print(f" Error: Source stage result missing method for {ds_type}: {e}")
            except Exception as e: print(f"  Error loading data for {ds_type}: {e}.")

        if total_rows == 0:
            print("Error: No rows found to process across all specified dataset types.")
            return (None, None, None)

        # 3. Process all collected rows sequentially with one progress bar
        print(f"\nProcessing {total_rows} rows sequentially...")
        processed, skipped, errors = 0, 0, 0
        # Single progress bar for all rows
        overall_progress_bar = tqdm(all_rows_data, total=total_rows, desc="Overall Progress", unit="row", leave=True)

        for row_dict in overall_progress_bar:
            dataset_type = row_dict['dataset_type'] # Get type injected earlier
            try:
                # Check cache
                cache_file = self._get_expected_cache_path(row_dict, dataset_type)
                if os.path.exists(cache_file):
                    skipped += 1
                    if skipped % 200 == 0: # Update less often for skips
                       overall_progress_bar.set_postfix(Processed=processed, Skipped=skipped, Errors=errors, refresh=False)
                    continue # Skip to next row

                # Process the row using the inner pipeline
                # verbose=False keeps stage logs quiet
                _ = self.inner_pipeline.run(data=row_dict.copy(), verbose=False) # Pass copy to stages
                processed += 1

            except Exception as e:
                errors += 1
                clip_info = row_dict.get('clip_folder', 'N/A')
                # Use tqdm.write to avoid messing up the bar
                overall_progress_bar.write(f"\n! Error processing row (clip: {clip_info}, type: {dataset_type}) ! Error: {e}")
                # Optional: Add full traceback here if needed for debugging
                # import traceback
                # overall_progress_bar.write(traceback.format_exc())

            # Update postfix on the progress bar
            if (processed + errors) % 20 == 0 or errors > 0: # Update somewhat regularly or if errors occur
                 overall_progress_bar.set_postfix(Processed=processed, Skipped=skipped, Errors=errors, refresh=False)


        overall_progress_bar.close() # Ensure the bar finishes cleanly

        # 4. Create DataLoaders after processing
        print("\n--- Creating DataLoaders ---")
        # Create loaders only for the types originally requested
        loaders = {}
        for ds_type in self.dataset_types_to_process:
             loaders[ds_type.lower()] = self.create_dataloader(ds_type)

        run_end_time = time.time()
        print(f"\n=== Sequential Pipeline Run Finished ({self.config_name} / v{self.pipeline_version}) | Total time: {(run_end_time - run_start_time):.2f}s ===")
        print(f"Final Counts: Processed={processed}, Skipped={skipped}, Errors={errors}")

        # Return loaders based on standard names
        return (loaders.get('train'), loaders.get('validation') or loaders.get('val'), loaders.get('test'))


# ----- Runner Function (Sequential Version) -----
def run_sequential_pipeline(config_path: str) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    Initializes and runs the ConfigurablePipeline sequentially.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
            DataLoaders for Train, Validation, and Test splits, or None.
    """
    train_loader, val_loader, test_loader = None, None, None # Default return values

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at '{config_path}'")
        return train_loader, val_loader, test_loader # Return None tuple

    print(f"\n--- Loading Config and Running Pipeline Sequentially from: {config_path} ---")
    try:
        # Instantiate the pipeline controller
        pipeline = ConfigurablePipeline(config_path=config_path)
        # Run the pipeline sequentially
        train_loader, val_loader, test_loader = pipeline.run_sequentially()

        # Brief summary after run completes (optional, as run_sequentially prints counts)
        # print("\n--- DataLoader Creation Summary ---")
        # if train_loader: print(f"Train DataLoader ready.")
        # if val_loader: print(f"Validation DataLoader ready.")
        # if test_loader: print(f"Test DataLoader ready.")

    except FileNotFoundError as e:
         print(f"\n!!! File not found during pipeline initialization: {e} !!!")
    except ValueError as e:
         print(f"\n!!! Configuration error: {e} !!!")
    except Exception as e:
         print(f"\n!!! Critical error during sequential pipeline execution: {e} !!!")
         import traceback; traceback.print_exc() # Show traceback for unexpected errors

    # Return the final state of the loaders
    return train_loader, val_loader, test_loader


# ----- Example Usage -----
if __name__ == "__main__":
    # No multiprocessing setup needed

    # Define config path
    config_file_path = "./configs/ENGAGENET_10fps_quality95_randdist.json" # Adjust path as needed

    train_dl, val_dl, test_dl = run_sequential_pipeline(config_file_path)

    print("\n--- Script Finished ---")
