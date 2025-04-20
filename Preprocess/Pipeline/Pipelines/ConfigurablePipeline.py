import os
import json
import importlib
import time
import math
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
# Lazy import DataLoader components later
# from torch.utils.data import DataLoader
from tqdm import tqdm
# --- Use Threading ---
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Project Imports ---
try:
    from Preprocess.Pipeline import config as project_config
except ImportError:
    project_config = None

# --- Assuming these imports exist in your project structure ---
try:
    # Only import base classes needed for initialization here
    from Preprocess.Pipeline.Encapsulation.CachedTensorDataset import CachedTensorDataset
    from Preprocess.Pipeline.OrchestrationPipeline import OrchestrationPipeline
    from Preprocess.Pipeline.Stages.SourceStage import SourceStage
except ImportError as e:
    print(f"Error importing core pipeline components: {e}. Please ensure they exist.")
    exit(1)


# ----- Worker Function for Chunks -----

def _process_chunk_thread_worker(
    chunk_data: List[Dict],
    pipeline_instance: 'ConfigurablePipeline'
) -> Dict:
    """
    Worker function executed by each thread. Processes a chunk of rows.
    Accesses the main pipeline instance directly.
    Returns aggregated results for the chunk.
    """
    processed_in_chunk = 0
    skipped_in_chunk = 0
    errors_in_chunk = 0
    error_details = [] # Collect error details for the chunk

    for row_dict in chunk_data:
        status = 'error'; error_msg = None
        clip_folder = str(row_dict.get('clip_folder', 'UnknownClip'))
        dataset_type = row_dict['dataset_type'] # Already injected

        try:
            # Calculate Cache Path using the pipeline instance's method
            cache_file = pipeline_instance._get_expected_cache_path(row_dict, dataset_type)

            if os.path.exists(cache_file):
                skipped_in_chunk += 1
                continue # Move to next row in chunk

            # Run Pipeline using the shared inner_pipeline instance
            # Pass a copy in case stages modify the dict in-place
            _ = pipeline_instance.inner_pipeline.run(data=row_dict.copy(), verbose=False)
            processed_in_chunk += 1
            status = 'processed'

        except Exception as e:
            import traceback
            errors_in_chunk += 1
            error_msg = f"Clip {clip_folder}: {type(e).__name__}: {e}" # Concise error for aggregation
            error_details.append(f"Error processing clip {clip_folder} (type: {dataset_type}):\n{traceback.format_exc()}")
            status = 'error'

    return {
        'processed': processed_in_chunk,
        'skipped': skipped_in_chunk,
        'errors': errors_in_chunk,
        'error_details': error_details # Pass detailed errors back if needed
    }


# ----- Updated ConfigurablePipeline Class -----

class ConfigurablePipeline:
    """ Runs preprocessing using Threading for row processing within each dataset type. """

    STAGE_COMMON_PARAMS = {
        "FrameExtractionStage": ["pipeline_version", "dataset_root", "cache_dir"],
        "TensorSavingStage": ["pipeline_version", "cache_dir", "config_name"],
    }

    # __init__ and _build_inner_pipeline, _get_expected_cache_path remain the same
    # as the previous threading version.
    def __init__(self, config_path: str = None, config_dict: Dict = None):
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

        source_stage_cfg = self.config.get('source_stage_config', {})
        self.source_stage = SourceStage(
            pipeline_version=self.pipeline_version, cache_dir=self.cache_root,
            metadata_csv_path=self.metadata_path, **source_stage_cfg
        )

        self.inner_pipeline = self._build_inner_pipeline(self.config.get('stages', []))
        self.data_loader_params = self.config.get('data_loader_params', {})
        self.dataset_types_to_process = self.config.get('dataset_types_to_process', ['Train', 'Validation', 'Test'])

        # print(f"Initialized Threaded Pipeline: '{self.config_name}' / v{self.pipeline_version} for '{self.dataset_name}'")

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
                 print(f"\n!!! Error loading/instantiating stage '{stage_name}' !!!")
                 print(f"Module Path: {module_path}, Params Passed: {params}")
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

    # --- Optimized method using chunking and threads ---
    def _process_dataframe_with_threads(
        self,
        df: pd.DataFrame,
        dataset_type: str,
        max_workers: Optional[int],
        overall_progress_bar: tqdm,
        chunk_size: int = 64 # Tunable parameter: Number of rows per thread task
    ) -> Tuple[int, int, int]:
        """ Processes DataFrame rows in chunks using ThreadPoolExecutor, updating an overall progress bar. """

        total_rows_in_df = len(df)
        if total_rows_in_df == 0: return 0, 0, 0

        # Determine number of workers - ensuring it's at least 1
        num_workers = max(1, max_workers or (os.cpu_count() * 2 if os.cpu_count() else 4))
        # Don't need more workers than chunks
        num_chunks = math.ceil(total_rows_in_df / chunk_size)
        num_workers = min(num_workers, num_chunks)

        processed_total, skipped_total, errors_total = 0, 0, 0

        # Prepare chunks of row dictionaries using faster itertuples
        chunks = []
        current_chunk = []
        # Get column names for creating dictionaries from tuples
        try:
            # df.columns includes the index name if it has one, handle this
            if df.index.name is not None:
                column_names = [df.index.name] + list(df.columns)
            else:
                 column_names = ['Index'] + list(df.columns) # Assign default index name if none

        except Exception as e:
            print(f"Error getting column names: {e}")
            # Fallback or re-raise, here we'll just use generic names
            column_names = [f'col_{i}' for i in range(len(next(df.itertuples())))]


        for row_tuple in df.itertuples(index=True, name='Row'): # Use named tuple for clarity if preferred
             # Convert tuple to dictionary - more memory but consistent with previous worker
             # Ensure column_names list matches the structure of row_tuple
             if len(column_names) != len(row_tuple):
                 print(f"Warning: Column name count ({len(column_names)}) doesn't match tuple length ({len(row_tuple)}). Skipping row.")
                 # Attempt to build dict anyway, might fail
                 try:
                     row_dict = dict(zip(column_names[:len(row_tuple)], row_tuple))
                 except: continue # Skip if zip fails
             else:
                row_dict = dict(zip(column_names, row_tuple))

             row_dict['dataset_type'] = dataset_type # Inject dataset type
             current_chunk.append(row_dict)
             if len(current_chunk) == chunk_size:
                 chunks.append(current_chunk)
                 current_chunk = []
        if current_chunk: # Add the last partial chunk
            chunks.append(current_chunk)

        # Process chunks using ThreadPoolExecutor
        futures = []
        with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix=f'{dataset_type}_Worker') as executor:
            # Submit chunk processing tasks
            for chunk in chunks:
                future = executor.submit(_process_chunk_thread_worker, chunk, self)
                futures.append(future)

            # Process results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    chunk_processed = result.get('processed', 0)
                    chunk_skipped = result.get('skipped', 0)
                    chunk_errors = result.get('errors', 0)
                    chunk_error_details = result.get('error_details', [])

                    processed_total += chunk_processed
                    skipped_total += chunk_skipped
                    errors_total += chunk_errors

                    # Update the OVERALL progress bar by the number of items processed/skipped in this chunk
                    items_in_chunk = chunk_processed + chunk_skipped + chunk_errors
                    overall_progress_bar.update(items_in_chunk)

                    # Report errors from the chunk
                    if chunk_errors > 0:
                        for err_detail in chunk_error_details:
                             # Limit length of printed tracebacks
                             overall_progress_bar.write(f"\n! Thread Error:\n{err_detail[:1000]}...")

                except Exception as e:
                    # Handle exceptions fetching results from futures (e.g., worker process died unexpectedly)
                    errors_total += chunk_size # Approximate chunk size as errors if future fails badly
                    overall_progress_bar.update(chunk_size) # Update bar by estimate
                    overall_progress_bar.write(f"\n! Critical error fetching result from thread chunk ! Error: {e}")

                # Update postfix on the overall bar less frequently
                if (processed_total + skipped_total + errors_total) % (chunk_size * 5) < chunk_size: # Update roughly every 5 chunks
                     overall_progress_bar.set_postfix(
                         Processed=processed_total, Skipped=skipped_total, Errors=errors_total, refresh=False
                     )

        # Return total counts for this dataset type
        return processed_total, skipped_total, errors_total


    # create_dataloader remains the same as previous threading version
    def create_dataloader(self, dataset_type: str) -> Optional[Any]: # Use Any for DataLoader
        try:
            import torch
            from torch.utils.data import DataLoader
            from Preprocess.Pipeline.Encapsulation.CachedTensorDataset import CachedTensorDataset
        except ImportError as e:
            print(f"Error: PyTorch/DataLoader components required but not found ({e})."); return None

        dl_params = self.data_loader_params
        batch_size = dl_params.get('batch_size', 32)
        dataloader_num_workers = dl_params.get('num_workers', 0)
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

    # --- Modified run method to use optimized thread processing ---
    def run_threaded(self, max_workers: Optional[int] = None, chunk_size: int = 64) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """ Run pipeline using threads processing rows in chunks with a single overall progress bar. """
        print(f"\n=== Starting Threaded Pipeline Run: '{self.config_name}' / v{self.pipeline_version} (Chunk Size: {chunk_size}) ===")
        run_start_time = time.time()

        # 1. Prepare dataset splits (same as before)
        print("Preparing dataset splits...")
        try:
            source_data = self.source_stage.process(verbose=False)
        except Exception as e:
            print(f"Error during source stage processing: {e}")
            return (None, None, None) # Cannot proceed

        # 2. Load dataframes and calculate total rows (same as before)
        dfs_to_process = []
        total_rows = 0
        print("Loading dataframes to calculate total size...")
        for ds_type in self.dataset_types_to_process:
            df = None
            try:
                if ds_type.lower() == 'train': df = source_data.get_train_data()
                elif ds_type.lower() in ['validation', 'val']: df = source_data.get_validation_data()
                elif ds_type.lower() == 'test': df = source_data.get_test_data()
                else: print(f"Warning: Unknown dataset type '{ds_type}' in config, skipping."); continue

                if df is not None and not df.empty:
                    dfs_to_process.append({'type': ds_type, 'df': df})
                    total_rows += len(df)
                    print(f"  Loaded {ds_type}: {len(df)} rows")
                else:
                    print(f"  Info: No data loaded for {ds_type}.")
            except FileNotFoundError: print(f"  Warning: Split CSV not found for {ds_type}.")
            except AttributeError as e: print(f" Error: Source stage result missing method for {ds_type}: {e}")
            except Exception as e: print(f"  Error loading data for {ds_type}: {e}.")

        if total_rows == 0:
            print("Error: No rows found to process across all specified dataset types.")
            return (None, None, None)

        # 3. Create the single overall progress bar (same as before)
        print(f"\nProcessing {total_rows} rows using threads...")
        overall_progress_bar = tqdm(total=total_rows, desc="Overall Progress", unit="row", leave=True)
        total_processed, total_skipped, total_errors = 0, 0, 0

        # 4. Process each dataframe sequentially, but use the *optimized* threaded method internally
        for data_info in dfs_to_process:
            ds_type = data_info['type']
            df = data_info['df']
            overall_progress_bar.set_description(f"Processing {ds_type}") # Update description for context
            # Call the optimized processing method for this specific dataframe
            processed, skipped, errors = self._process_dataframe_with_threads(
                df, ds_type, max_workers, overall_progress_bar, chunk_size
            )
            # Accumulate totals
            total_processed += processed
            total_skipped += skipped
            total_errors += errors
            # Ensure final postfix for this dataset is shown on the overall bar
            overall_progress_bar.set_postfix(Processed=total_processed, Skipped=total_skipped, Errors=total_errors, refresh=True)

        overall_progress_bar.close() # Close the main bar

        # 5. Create DataLoaders (same as before)
        print("\n--- Creating DataLoaders ---")
        loaders = {}
        for ds_type in self.dataset_types_to_process:
            loaders[ds_type.lower()] = self.create_dataloader(ds_type)

        run_end_time = time.time()
        print(f"\n=== Threaded Pipeline Run Finished ({self.config_name} / v{self.pipeline_version}) | Total time: {(run_end_time - run_start_time):.2f}s ===")
        print(f"Final Counts: Processed={total_processed}, Skipped={total_skipped}, Errors={total_errors}")

        # Return loaders based on standard names
        return (loaders.get('train'), loaders.get('validation') or loaders.get('val'), loaders.get('test'))


# ----- Runner Function -----
# Renamed run method in ConfigurablePipeline to run_threaded
def run_threaded_pipeline(
    config_path: str,
    max_workers: Optional[int] = None, # Allow None for default calculation
    chunk_size: int = 64 # Make chunk size configurable here
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """ Run the pipeline with threading support using row chunking. """
    train_loader, val_loader, test_loader = None, None, None

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at '{config_path}'")
        # Use exit() or return based on desired behavior for missing config
        return train_loader, val_loader, test_loader # Return None tuple

    print(f"\n--- Loading Config and Running Threaded Pipeline from: {config_path} ---")
    try:
        pipeline = ConfigurablePipeline(config_path=config_path)
        # Call the correctly named run method, passing chunk_size
        train_loader, val_loader, test_loader = pipeline.run_threaded(
            max_workers=max_workers,
            chunk_size=chunk_size
        )

        # Optional: Print summary of created loaders
        print("\n--- DataLoader Creation Summary ---")
        if train_loader: print(f"Train DataLoader ready.") # Length check can be slow if dataset is large
        else: print("Train DataLoader: None.")
        if val_loader: print(f"Validation DataLoader ready.")
        else: print("Validation DataLoader: None.")
        if test_loader: print(f"Test DataLoader ready.")
        else: print("Test DataLoader: None.")

    except FileNotFoundError as e:
         print(f"\n!!! File not found during pipeline initialization: {e} !!!")
    except ValueError as e:
         print(f"\n!!! Configuration error: {e} !!!")
    except Exception as e:
         print(f"\n!!! Critical error during pipeline execution: {e} !!!")
         import traceback; traceback.print_exc()

    return train_loader, val_loader, test_loader


# ----- Example Usage -----
if __name__ == "__main__":
    # No multiprocessing setup needed

    config_file_path = "./configs/ENGAGENET_10fps_quality95_randdist.json" # Or your DAISEE config path

    # --- Configuration for Threading ---
    # Let pipeline calculate default based on CPU cores, or set manually
    num_workers = None # Set to None to use default (os.cpu_count() * 2) or e.g., 8, 16 etc.
    # Size of row chunks processed by each thread task. Larger chunks reduce overhead but decrease parallelism granularity.
    # Tune based on typical row processing time and task overhead. Start with 64 or 128.
    row_chunk_size = 128

    print(f"Running threaded pipeline with max_workers={num_workers or 'Default'}, chunk_size={row_chunk_size}")

    # Call the runner function
    run_threaded_pipeline(
        config_file_path,
        max_workers=num_workers,
        chunk_size=row_chunk_size
    )

    print("\n--- Script Finished ---")