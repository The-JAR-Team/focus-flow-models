import os
import time
from torch.utils.data import DataLoader, Dataset

# Import your configuration constants.
from Preprocess.Pipeline.DaiseeConfig import CACHE_DIR
from Preprocess.Pipeline.Encapsulation.CachedTensorDataset import CachedTensorDataset
# Import stages.
from Preprocess.Pipeline.Stages.SourceStage import SourceStage
from Preprocess.Pipeline.Stages.FrameExtractionStage import FrameExtractionStage
from Preprocess.Pipeline.Stages.MediapipeProcessingStage import MediapipeProcessingStage
from Preprocess.Pipeline.Stages.TensorStackingStage import TensorStackingStage
from Preprocess.Pipeline.Stages.TensorSavingStage import TensorSavingStage
from Preprocess.Pipeline.OrchestrationPipeline import OrchestrationPipeline

class SimplePipeline:
    """
    Main pipeline that orchestrates the entire preprocessing flow:
      1. Run the SourceStage to obtain CSVs.
      2. For each dataset (Train, Validation, Test), for each row:
           a. Check if a cached tensor result exists in the proper cache directory.
           b. If not, run the inner pipeline (FrameExtractionStage, MediapipeProcessingStage, TensorStackingStage, TensorSavingStage)
              to generate and save a fixed-size tensor stack.
      3. Create DataLoaders for each dataset from the cache.
    """
    def __init__(self, pipeline_version: str):
        self.pipeline_version = pipeline_version
        self.cache_root = CACHE_DIR  # Use CACHE_DIR from DaiseeConfig.py

        self.source_stage = SourceStage(pipeline_version)
        # Build the inner pipeline stages:
        frame_extraction_stage = FrameExtractionStage(
            pipeline_version=pipeline_version,
            save_frames=False,  # Set to True if you want to save frames to disk.
            desired_fps=24.0,
            jpeg_quality=50
        )
        mediapipe_stage = MediapipeProcessingStage()
        tensor_stacking_stage = TensorStackingStage(target_frames=240, num_landmarks=478, dims=3)
        tesnor_saving_stage = TensorSavingStage(pipeline_version=pipeline_version,
                                                cache_root=self.cache_root)
        # For each dataset, TensorSavingStage will be instantiated with that dataset type.
        # Create an inner pipeline that runs the four stages in sequence.
        self.inner_pipeline = OrchestrationPipeline(
            stages=[frame_extraction_stage, mediapipe_stage, tensor_stacking_stage, tesnor_saving_stage]
        )
        # Note: We'll call TensorSavingStage separately within process_dataset.

    def process_dataset(self, dataset_type: str) -> None:
        """
        Process all rows for the given dataset type (e.g., 'Train') and save tensor results to cache.
        Prints progress every 10 rows, including percentage complete and timing statistics.
        Every 100 rows, runs the inner pipeline with verbose=True.
        Before processing a row, checks if its corresponding cache file already exists.
        """
        # Load the appropriate CSV from SourceStage.
        source_data = self.source_stage.process(verbose=False)
        if dataset_type.lower() == 'train':
            df = source_data.get_train_data()
        elif dataset_type.lower() == 'validation':
            df = source_data.get_validation_data()
        elif dataset_type.lower() == 'test':
            df = source_data.get_test_data()
        else:
            raise ValueError("dataset_type must be 'Train', 'Validation', or 'Test'.")

        total_rows = len(df)
        print(f"Processing {total_rows} rows for {dataset_type} dataset...")

        counter = 0.0
        print_idx = 10
        import time  # Ideally, import at the top.
        for idx, row in df.iterrows():
            # Compute expected cache file path based on the row's clip_folder.
            clip_folder = str(row['clip_folder'])
            subfolder = clip_folder[:6]
            cache_file = os.path.join(self.cache_root, "PipelineResult", self.pipeline_version, dataset_type, subfolder,
                                      f"{clip_folder}_{self.pipeline_version}.pt")
            # If the cache file exists, skip processing this row.
            if os.path.exists(cache_file):
                print(f"Row {idx + 1}: Already processed for clip {clip_folder}. Skipping.")
                continue

            start_time = time.time()
            # Use verbose=True every 100 rows.
            if (idx + 1) % 100 == 0:
                self.inner_pipeline.run(data=row, verbose=True)
            else:
                self.inner_pipeline.run(data=row, verbose=False)
            # Now, save the result (TensorSavingStage will use clip_folder from the result).
            end_time = time.time()
            row_time = end_time - start_time
            counter += row_time

            # Print progress every 10 rows or on the final row.
            if (idx + 1) % print_idx == 0 or (idx + 1) == total_rows:
                avg_time = counter / print_idx
                percentage = 100.0 * (idx + 1) / total_rows
                print(f"Processed {idx + 1}/{total_rows} rows ({percentage:.2f}%). "
                      f"Avg time per row: {avg_time:.2f}s. Last 10 rows took: {counter:.2f}s.")
                counter = 0.0

    def create_dataloader(self, dataset_type: str, batch_size: int = 32) -> DataLoader:
        """
        Creates a DataLoader for the given dataset type by recursively loading the cached tensor results.
        """
        # The root directory for cached tensor results for this dataset:
        cache_dir = os.path.join(self.cache_root, "PipelineResult", self.pipeline_version, dataset_type)
        dataset = CachedTensorDataset(cache_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(dataset_type.lower() == 'train'))
        return dataloader

    def run(self):
        # Process each dataset type.
        for ds in ['Train', 'Validation', 'Test']:
            self.process_dataset(ds)
        # Create DataLoaders.
        train_loader = self.create_dataloader('Train')
        val_loader = self.create_dataloader('Validation')
        test_loader = self.create_dataloader('Test')
        return train_loader, val_loader, test_loader


# ----- Example Usage -----
if __name__ == "__main__":
    pipeline_version = "01"
    main_pipeline = SimplePipeline(pipeline_version=pipeline_version)
    train_loader, val_loader, test_loader = main_pipeline.run()

    # Demonstrate DataLoader content for the Train dataset.
    print("Final DataLoader for Train dataset. Sample contents:")
    for tensor_stack, label in train_loader:
        print("Tensor stack shape:", tensor_stack.shape)
        print("Label:", label)
        break
