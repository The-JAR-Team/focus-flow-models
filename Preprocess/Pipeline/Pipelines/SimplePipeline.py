import os
import time
from torch.utils.data import DataLoader
from Preprocess.Pipeline import config
from Preprocess.Pipeline.Encapsulation.CachedTensorDataset import CachedTensorDataset
from Preprocess.Pipeline.Stages.SourceStage import SourceStage
from Preprocess.Pipeline.Stages.FrameExtractionStage import FrameExtractionStage
from Preprocess.Pipeline.Stages.MediapipeProcessingStage import MediapipeProcessingStage
from Preprocess.Pipeline.Stages.TensorStackingStage import TensorStackingStage
from Preprocess.Pipeline.Stages.TensorSavingStage import TensorSavingStage
from Preprocess.Pipeline.OrchestrationPipeline import OrchestrationPipeline

DAISEE_METADATA_PATH = r"../MetaData/daisee_metadata.csv"
ENGAGENET_METADATA_PATH = r"../MetaData/engagenet_metadata.csv"


class SimplePipeline:
    """
    Main pipeline orchestrating preprocessing for a specified dataset (DAiSEE or EngageNet).
    Uses configurable stages and paths.
    """
    def __init__(self, pipeline_version: str, dataset_name: str,
                 # Add stage-specific configs here or load from a file
                 source_config: dict = None,
                 frame_config: dict = None,
                 # mediapipe_config: dict = None, # Example if needed
                 stacking_config: dict = None,
                 saving_config: dict = None):
        """
        Initializes the pipeline for a specific dataset.

        Args:
            pipeline_version (str): Version string for caching outputs.
            dataset_name (str): Name of the dataset to process ('DAiSEE' or 'EngageNet').
            source_config (dict, optional): Configuration for SourceStage.
            frame_config (dict, optional): Configuration for FrameExtractionStage.
            stacking_config (dict, optional): Configuration for TensorStackingStage.
            saving_config (dict, optional): Configuration for TensorSavingStage.
        """
        self.pipeline_version = pipeline_version
        self.dataset_name = dataset_name
        self.cache_root = config.CACHE_DIR # Use CACHE_DIR from config.py

        # --- Determine dataset-specific paths ---
        if dataset_name.upper() == 'DAISEE':
            self.dataset_root = config.DAISEE_DATASET_ROOT
            self.metadata_path = DAISEE_METADATA_PATH
            print(f"Initializing pipeline for DAiSEE dataset.")
        elif dataset_name.upper() == 'ENGAGENET':
            self.dataset_root = config.ENGAGENET_DATASET_ROOT
            self.metadata_path = ENGAGENET_METADATA_PATH
            print(f"Initializing pipeline for EngageNet dataset.")
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}. Use 'DAiSEE' or 'EngageNet'.")

        if not os.path.exists(self.metadata_path):
             print(f"CRITICAL WARNING: Metadata file not found for {dataset_name} at: {self.metadata_path}")
             # Consider raising an error here depending on desired behavior

        # --- Configure Stages (using defaults if no config provided) ---
        source_cfg = source_config or {}
        frame_cfg = frame_config or {}
        stacking_cfg = stacking_config or {}
        # saving_cfg = saving_config or {} # Currently only needs cache_root + pipeline_version

        self.source_stage = SourceStage(
            pipeline_version=pipeline_version,
            cache_dir=self.cache_root,
            metadata_csv_path=self.metadata_path,
            perform_random_split=source_cfg.get('perform_random_split', True), # Default: use column split
            random_seed=source_cfg.get('random_seed', 42),
            train_ratio=source_cfg.get('train_ratio', 0.6),
            val_ratio=source_cfg.get('val_ratio', 0.2),
            test_ratio=source_cfg.get('test_ratio', 0.2),
            stratify_column=source_cfg.get('stratify_column', None) # e.g., 'engagement_label' for EngageNet
        )

        frame_extraction_stage = FrameExtractionStage(
            pipeline_version=pipeline_version,
            dataset_root=self.dataset_root, # Pass correct dataset root
            cache_dir=self.cache_root,
            save_frames=frame_cfg.get('save_frames', False), # Don't save frames by default?
            desired_fps=frame_cfg.get('desired_fps', 10.0),
            jpeg_quality=frame_cfg.get('jpeg_quality', 95),
            resize_width=frame_cfg.get('resize_width', None),
            resize_height=frame_cfg.get('resize_height', None)
        )

        mediapipe_stage = MediapipeProcessingStage() # Add config if needed

        tensor_stacking_stage = TensorStackingStage(
            target_frames=stacking_cfg.get('target_frames', 240), # Example config
            num_landmarks=stacking_cfg.get('num_landmarks', 478), # Example config
            dims=stacking_cfg.get('dims', 3) # Example config
        )

        # TensorSavingStage correctly receives cache_root and version
        tensor_saving_stage = TensorSavingStage(
            pipeline_version=pipeline_version,
            cache_root=self.cache_root,
            config_name='testtest'
            # Add dataset_type or other configs if needed by TensorSavingStage logic
        )

        # --- Define Inner Pipeline ---
        # Note: TensorSavingStage might need special handling if its run depends
        # on the output of the previous stage within the orchestrator.
        # The original code called inner_pipeline.run *without* saving stage,
        # then saved separately. Let's keep that pattern for now.
        self.inner_pipeline = OrchestrationPipeline(
            stages=[frame_extraction_stage, mediapipe_stage, tensor_stacking_stage] # Exclude saving stage here
        )
        self.tensor_saving_stage = tensor_saving_stage # Keep saving stage separate

    def process_dataset(self, dataset_type: str) -> None:
        """
        Process all rows for the given dataset type (e.g., 'Train')
        and save tensor results to cache.
        """
        print(f"\n--- Processing dataset: {dataset_type} ---")
        # Load the appropriate CSV from SourceStage.
        # SourceStage handles caching of the split CSVs itself.
        source_data = self.source_stage.process(verbose=False) # Run source stage (gets cached/creates splits)

        if dataset_type.lower() == 'train':
            df = source_data.get_train_data() # Assumes SourceData has these methods
        elif dataset_type.lower() in ['validation', 'val']: # Accept both 'val' and 'validation'
            df = source_data.get_validation_data()
        elif dataset_type.lower() == 'test':
            df = source_data.get_test_data()
        else:
            raise ValueError("dataset_type must be 'Train', 'Validation', or 'Test'.")

        if df is None or df.empty:
             print(f"Warning: No data found for dataset type '{dataset_type}'. Skipping processing.")
             return

        total_rows = len(df)
        print(f"Processing {total_rows} rows for {dataset_type} dataset...")

        processed_count = 0
        skipped_count = 0
        error_count = 0
        start_batch_time = time.time()
        rows_in_batch = 0
        BATCH_PRINT_INTERVAL = 50 # Print progress every N rows

        for idx, row_tuple in enumerate(df.iterrows()):
            row_index, row = row_tuple # iterrows() yields (index, Series)
            row_dict = row.to_dict() # Convert row Series to dict for stages

            # --- Check Cache for FINAL Tensor Result ---
            # Use the unified 'clip_folder' and 'person' columns
            clip_folder = str(row_dict['clip_folder'])
            person_id = str(row_dict['person']) # Use person_id for subfolder structure? Or clip_folder[:6]? Adapt as needed.

            # Define final cache path (adjust subfolder logic if needed)
            # Example using person_id as subfolder:
            # subfolder = person_id
            # Example using first 6 chars of clip_folder (like original):
            subfolder = clip_folder[:6] if len(clip_folder) >= 6 else clip_folder

            cache_file = os.path.join(
                self.cache_root,
                "PipelineResult", # Standardized output folder
                self.pipeline_version, # Version specific
                dataset_type, # Train/Val/Test
                subfolder, # Subfolder structure
                f"{clip_folder}_{self.pipeline_version}.pt" # Final tensor file
            )

            if os.path.exists(cache_file):
                skipped_count += 1
                # Optionally print less frequently for skips
                # if skipped_count % 100 == 0: print(f"Row {idx + 1} ({clip_folder}): Cached result exists. Skipping.")
                continue # Skip to next row if cached

            # --- Process Row ---
            rows_in_batch += 1
            try:
                # Run the inner pipeline (extract -> mediapipe -> stack)
                # Verbosity can be controlled here
                verbose_run = (idx + 1) % 100 == 0 # Verbose every 100 rows
                # print(f"DEBUG: Running inner pipeline for row {idx+1}, clip: {clip_folder}") # Debug print
                pipeline_result = self.inner_pipeline.run(data=row_dict, verbose=verbose_run)

                # Explicitly run the saving stage with the result
                # Pass dataset_type if saving stage needs it for path construction
                self.tensor_saving_stage.process(pipeline_result, verbose=verbose_run)

                processed_count += 1

            except Exception as e:
                error_count += 1
                print(f"\n!!! Error processing row {idx + 1} (clip: {clip_folder}) !!!")
                print(f"Row Data: {row_dict}")
                print(f"Error: {e}")
                # Decide whether to continue or stop on error
                # continue

            # --- Print Progress ---
            if (idx + 1) % BATCH_PRINT_INTERVAL == 0 or (idx + 1) == total_rows:
                end_batch_time = time.time()
                batch_time = end_batch_time - start_batch_time
                time_per_row = batch_time / rows_in_batch if rows_in_batch > 0 else 0
                percentage = 100.0 * (idx + 1) / total_rows
                print(f"  Processed {idx + 1}/{total_rows} rows ({percentage:.2f}%). "
                      f"Current Batch ({rows_in_batch} rows): {batch_time:.2f}s ({time_per_row:.3f}s/row). "
                      f"Total Processed: {processed_count}, Skipped(Cached): {skipped_count}, Errors: {error_count}")
                start_batch_time = time.time() # Reset timer
                rows_in_batch = 0 # Reset counter

        print(f"--- Finished processing {dataset_type} dataset ---")
        print(f"  Total Processed: {processed_count}, Skipped(Cached): {skipped_count}, Errors: {error_count}")


    def create_dataloader(self, dataset_type: str, batch_size: int = 32) -> DataLoader:
        """
        Creates a DataLoader for the given dataset type from cached tensor results.
        """
        print(f"\n--- Creating DataLoader for: {dataset_type} ---")
        # Root directory for the final cached tensors for this dataset/version
        cache_dir = os.path.join(self.cache_root, "PipelineResult", self.pipeline_version, dataset_type)

        if not os.path.isdir(cache_dir):
             print(f"Warning: Cache directory for final tensors not found: {cache_dir}")
             print("DataLoader will be empty.")
             return None # Or return an empty DataLoader

        # Assuming CachedTensorDataset recursively finds .pt files and loads them
        try:
            dataset = CachedTensorDataset(cache_dir)
            if len(dataset) == 0:
                 print(f"Warning: No .pt files found in cache directory: {cache_dir}")
                 return None # Or return empty DataLoader
            print(f"Found {len(dataset)} cached samples for {dataset_type} dataset.")
            # Shuffle only for the training set
            should_shuffle = (dataset_type.lower() == 'train')
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=should_shuffle, num_workers=4, pin_memory=True) # Added workers/pin_memory
            return dataloader
        except Exception as e:
            print(f"Error creating CachedTensorDataset or DataLoader for {dataset_type} from {cache_dir}: {e}")
            return None

    def run(self):
        """
        Runs the full pipeline: process datasets and create DataLoaders.
        """
        print(f"=== Starting Pipeline Run: version='{self.pipeline_version}', dataset='{self.dataset_name}' ===")
        start_run_time = time.time()

        # Process each dataset type
        for ds in ['Train', 'Validation', 'Test']:
            self.process_dataset(ds)

        # Create DataLoaders
        train_loader = self.create_dataloader('Train')
        val_loader = self.create_dataloader('Validation')
        test_loader = self.create_dataloader('Test')

        end_run_time = time.time()
        print(f"\n=== Pipeline Run Finished ===")
        print(f"Total time: {(end_run_time - start_run_time):.2f} seconds.")
        print(f"Train DataLoader ready: {'Yes' if train_loader else 'No'}")
        print(f"Validation DataLoader ready: {'Yes' if val_loader else 'No'}")
        print(f"Test DataLoader ready: {'Yes' if test_loader else 'No'}")

        return train_loader, val_loader, test_loader


# ----- Example Usage -----
if __name__ == "__main__":
    # --- Configuration ---
    PIPELINE_VERSION = "EngageNet_v01" # Example version string
    DATASET_TO_PROCESS = "EngageNet" # Choose 'DAiSEE' or 'EngageNet'

    # Optional: Define specific stage configurations
    source_cfg = {
        "perform_random_split": True, # Use 'subset' column by default
        "stratify_column": "engagement_label" # Used only if perform_random_split is True
    }
    frame_cfg = {
        "desired_fps": 15, # Lower FPS
        "save_frames": False
    }
    stacking_cfg = {
        "target_frames": 150 # Adjusted based on potential lower FPS * duration
    }

    # --- Run ---
    main_pipeline = SimplePipeline(
        pipeline_version=PIPELINE_VERSION,
        dataset_name=DATASET_TO_PROCESS,
        source_config=source_cfg,
        frame_config=frame_cfg,
        stacking_config=stacking_cfg
    )
    train_loader, val_loader, test_loader = main_pipeline.run()

    # Demonstrate DataLoader content (if loaders were created successfully)
    if train_loader:
        print("\n--- Sample from Train DataLoader ---")
        try:
            for i, (tensor_stack, label) in enumerate(train_loader):
                print(f"Batch {i+1}:")
                print("  Tensor stack shape:", tensor_stack.shape) # Shape: [batch_size, target_frames, num_landmarks*dims] or similar
                # Labels will be complex dicts from CachedTensorDataset - need to inspect its format
                # Assuming it returns the label dict directly:
                print(f"  Label batch size: {len(label.get('engagement_numeric', [])) if isinstance(label, dict) else 'N/A'}") # Example access
                print(f"  Example Label Dict in Batch: { {k: v[0] if hasattr(v, '__getitem__') else v for k, v in label.items()} if isinstance(label, dict) else label }") # Print first item's label dict
                break # Only show first batch
        except Exception as e:
            print(f"Error iterating through train_loader: {e}")
    else:
        print("\nTrain DataLoader was not created.")
