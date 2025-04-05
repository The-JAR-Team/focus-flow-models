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
        # For each dataset, TensorSavingStage will be instantiated with that dataset type.
        # Create an inner pipeline that runs the four stages in sequence.
        self.inner_pipeline = OrchestrationPipeline(
            stages=[frame_extraction_stage, mediapipe_stage, tensor_stacking_stage]
        )
        # Note: We'll call TensorSavingStage separately within process_dataset.

    def process_dataset(self, dataset_type: str) -> None:
        """
        Process all rows for the given dataset type (e.g., 'Train') and save tensor results to cache.
        Prints progress every 10 rows, including percentage complete and timing statistics.
        Every 100 rows, runs the inner pipeline with verbose=True.
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

        # Instantiate a TensorSavingStage for this dataset.
        save_stage = TensorSavingStage(pipeline_version=self.pipeline_version,
                                       cache_root=self.cache_root)

        cumulative_time = 0.0
        last_10_time = 0.0
        for idx, row in df.iterrows():
            start_time = time.time()
            # Use verbose=True every 100 rows.
            if (idx + 1) % 100 == 0:
                # Run inner pipeline, then pass result to saving stage.
                result = self.inner_pipeline.run(data=row, verbose=True)
            else:
                result = self.inner_pipeline.run(data=row, verbose=False)
            # Now, call the saving stage on the result.
            # The saving stage now uses the clip_folder from the result.
            save_stage.process(result, verbose=False)
            end_time = time.time()
            row_time = end_time - start_time
            cumulative_time += row_time
            last_10_time += row_time

            # Print progress every 10 rows or on the final row.
            if (idx + 1) % 10 == 0 or (idx + 1) == total_rows:
                avg_time = cumulative_time / (idx + 1)
                percentage = 100.0 * (idx + 1) / total_rows
                print(f"Processed {idx + 1}/{total_rows} rows ({percentage:.2f}%). "
                      f"Avg time per row: {avg_time:.2f}s. Last 10 rows took: {last_10_time:.2f}s.")
                last_10_time = 0.0

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
