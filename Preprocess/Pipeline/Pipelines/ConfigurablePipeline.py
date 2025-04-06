import os
import json
import importlib
import time
from typing import Dict, List, Any, Tuple
from torch.utils.data import DataLoader

# Import your configuration constants
from Preprocess.Pipeline.DaiseeConfig import CACHE_DIR
from Preprocess.Pipeline.Encapsulation.CachedTensorDataset import CachedTensorDataset
# Import the orchestration pipeline
from Preprocess.Pipeline.OrchestrationPipeline import OrchestrationPipeline
# Import source stage
from Preprocess.Pipeline.Stages.SourceStage import SourceStage


class ConfigurablePipeline:
    """
    A configurable pipeline that can be initialized from a JSON configuration file.
    The pipeline orchestrates the entire preprocessing flow:
      1. Run the SourceStage to obtain CSVs.
      2. For each dataset (Train, Validation, Test), for each row:
           a. Check if a cached tensor result exists in the proper cache directory.
           b. If not, run the inner pipeline with dynamically loaded stages to generate and save a fixed-size tensor stack.
      3. Create DataLoaders for each dataset from the cache.
    """

    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """
        Initialize the configurable pipeline from either a JSON file path or a dictionary.

        Args:
            config_path: Path to the JSON configuration file
            config_dict: Dictionary containing the configuration

        Note:
            pipeline_version: Used to track compatibility between different versions of pipeline code
            config_name: Used to identify different preprocessing configurations/settings
        """
        if config_path is not None:
            # Load configuration from JSON file
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                # Extract config name from file path if not defined in config
                if 'config_name' not in self.config:
                    self.config['config_name'] = os.path.basename(config_path).split('.')[0]
        elif config_dict is not None:
            # Use provided configuration dictionary
            self.config = config_dict
            # Ensure config_name exists
            if 'config_name' not in self.config:
                self.config['config_name'] = 'default_config'
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        # Extract configuration parameters
        self.pipeline_version = self.config.get('pipeline_version', '01')
        self.config_name = self.config.get('config_name')
        self.cache_root = CACHE_DIR
        self.dataset_types = self.config.get('dataset_types', ['Train', 'Validation', 'Test'])

        # Initialize the source stage
        self.source_stage = SourceStage(self.pipeline_version)

        # Build the inner pipeline with dynamic stage loading
        self.inner_pipeline = self._build_inner_pipeline()

    def _build_inner_pipeline(self) -> OrchestrationPipeline:
        """
        Dynamically build the inner pipeline based on the configuration.

        Returns:
            An OrchestrationPipeline instance with the configured stages
        """
        stages = []
        stage_configs = self.config.get('stages', [])

        for stage_config in stage_configs:
            stage_name = stage_config.get('name')
            params = stage_config.get('params', {})

            # Add pipeline_version to params if the stage might need it
            if stage_name in ['FrameExtractionStage', 'TensorSavingStage']:
                params['pipeline_version'] = self.pipeline_version

            # Add config_name and cache_root to TensorSavingStage
            if stage_name == 'TensorSavingStage':
                params['cache_root'] = self.cache_root
                params['config_name'] = self.config_name

            # Dynamically import and instantiate the stage class
            try:
                # Import directly from the Stages package - this matches your folder structure
                module_path = f"Preprocess.Pipeline.Stages.{stage_name}"
                print(f"Attempting to import {stage_name} from {module_path}")

                # Direct imports for each stage type
                if stage_name == "FrameExtractionStage":
                    from Preprocess.Pipeline.Stages.FrameExtractionStage import FrameExtractionStage
                    stage_instance = FrameExtractionStage(**params)
                elif stage_name == "MediapipeProcessingStage":
                    from Preprocess.Pipeline.Stages.MediapipeProcessingStage import MediapipeProcessingStage
                    stage_instance = MediapipeProcessingStage(**params)
                elif stage_name == "SourceStage":
                    from Preprocess.Pipeline.Stages.SourceStage import SourceStage
                    stage_instance = SourceStage(**params)
                elif stage_name == "TensorSavingStage":
                    from Preprocess.Pipeline.Stages.TensorSavingStage import TensorSavingStage
                    stage_instance = TensorSavingStage(**params)
                elif stage_name == "TensorStackingStage":
                    from Preprocess.Pipeline.Stages.TensorStackingStage import TensorStackingStage
                    stage_instance = TensorStackingStage(**params)
                else:
                    # Fallback to dynamic import for any other stages
                    try:
                        module = importlib.import_module(module_path)
                        stage_class = getattr(module, stage_name)
                        stage_instance = stage_class(**params)
                    except (ImportError, AttributeError) as e:
                        print(f"Dynamic import failed: {e}")
                        raise

                stages.append(stage_instance)
            except Exception as e:
                print(f"Error loading stage {stage_name}: {e}")
                print(f"Params passed to {stage_name}: {params}")
                raise ValueError(f"Failed to load stage {stage_name}: {e}")

        return OrchestrationPipeline(stages=stages)

    def process_dataset(self, dataset_type: str) -> None:
        """
        Process all rows for the given dataset type and save tensor results to cache.
        Prints progress every 10 rows, including percentage complete and timing statistics.

        Args:
            dataset_type: The type of dataset to process ('Train', 'Validation', or 'Test')
        """
        # Load the appropriate CSV from SourceStage
        source_data = self.source_stage.process(verbose=False)
        if dataset_type.lower() == 'train':
            df = source_data.get_train_data()
        elif dataset_type.lower() == 'validation':
            df = source_data.get_validation_data()
        elif dataset_type.lower() == 'test':
            df = source_data.get_test_data()
        else:
            raise ValueError(f"dataset_type must be one of {self.dataset_types}")

        total_rows = len(df)
        print(f"Processing {total_rows} rows for {dataset_type} dataset...")

        counter = 0.0
        print_idx = 10

        for idx, row in df.iterrows():
            # Compute expected cache file path based on the row's clip_folder
            clip_folder = str(row['clip_folder'])
            subfolder = clip_folder[:6]
            cache_file = os.path.join(self.cache_root, "PipelineResult", self.config_name, self.pipeline_version,
                                      dataset_type, subfolder, f"{clip_folder}_{self.pipeline_version}.pt")

            # If the cache file exists, skip processing this row
            if os.path.exists(cache_file):
                print(f"Row {idx + 1}: Already processed for clip {clip_folder}. Skipping.")
                continue

            start_time = time.time()
            # Use verbose=True every 100 rows
            verbose = (idx + 1) % 100 == 0
            self.inner_pipeline.run(data=row, verbose=verbose)

            end_time = time.time()
            row_time = end_time - start_time
            counter += row_time

            # Print progress every 10 rows or on the final row
            if (idx + 1) % print_idx == 0 or (idx + 1) == total_rows:
                avg_time = counter / min(print_idx, idx % print_idx + 1 if idx % print_idx != 0 else print_idx)
                percentage = 100.0 * (idx + 1) / total_rows
                print(f"Processed {idx + 1}/{total_rows} rows ({percentage:.2f}%). "
                      f"Avg time per row: {avg_time:.2f}s. Last batch took: {counter:.2f}s.")
                counter = 0.0

    def create_dataloader(self, dataset_type: str, batch_size: int = 32) -> DataLoader:
        """
        Creates a DataLoader for the given dataset type by recursively loading the cached tensor results.

        Args:
            dataset_type: The type of dataset to create a DataLoader for
            batch_size: The batch size for the DataLoader

        Returns:
            A DataLoader instance for the specified dataset
        """
        # The root directory for cached tensor results for this dataset
        cache_dir = os.path.join(self.cache_root, "PipelineResult", self.config_name, self.pipeline_version,
                                 dataset_type)
        dataset = CachedTensorDataset(cache_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(dataset_type.lower() == 'train'))
        return dataloader

    def run(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Run the pipeline for all configured dataset types and create DataLoaders.

        Returns:
            A tuple of DataLoaders for train, validation, and test datasets
        """
        # Process each dataset type
        for ds in self.dataset_types:
            self.process_dataset(ds)

        # Create DataLoaders
        loaders = {}
        for ds in self.dataset_types:
            loaders[ds.lower()] = self.create_dataloader(ds)

        return (loaders.get('train'), loaders.get('validation'), loaders.get('test'))


def load_pipeline_from_json(config_path: str) -> ConfigurablePipeline:
    """
    Helper function to create a pipeline from a JSON configuration file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        A ConfigurablePipeline instance
    """
    return ConfigurablePipeline(config_path=config_path)


# ----- Example Usage -----
if __name__ == "__main__":
    # Example 1: Load from a JSON file
    config_path = "configs/01_24fps_quality50.json"
    pipeline = load_pipeline_from_json(config_path)
    train_loader, val_loader, test_loader = pipeline.run()

    # Demonstrate DataLoader content for the Train dataset
    print("Final DataLoader for Train dataset. Sample contents:")
    for tensor_stack, label in train_loader:
        print("Tensor stack shape:", tensor_stack.shape)
        print("Label:", label)
        break