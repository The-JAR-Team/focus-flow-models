import os
import json
from typing import List, Dict, Optional, Tuple, Union

from Model_Training.data_handling.dataset import CachedTensorDataset
from Model_Training.pipelines.pipeline import OrchestrationPipeline


def load_data_sources_config(json_path: str) -> Dict:
    """Loads the data sources configuration from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            config_data = json.load(f)
            if "dataset_configurations" not in config_data:
                raise ValueError("'dataset_configurations' key missing in data sources JSON.")
            return config_data
    except FileNotFoundError:
        print(f"Error: Data sources JSON file not found at {json_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        raise
    except ValueError as ve:
        print(f"Error in data sources JSON structure: {ve}")
        raise


def get_hf_datasets(
        dataset_config_name: str,
        data_sources_json_path: str,
        train_pipeline: OrchestrationPipeline,
        val_pipeline: OrchestrationPipeline,
        test_pipeline: OrchestrationPipeline,
        verbose: bool = False
) -> Dict[str, Optional[CachedTensorDataset]]:
    """
    Creates train, validation, and test CachedTensorDataset instances.
    Pipelines are now passed as arguments.

    Args:
        dataset_config_name (str): Key for the dataset configuration in data_sources.json.
        data_sources_json_path (str): Path to the data_sources.json file.
        train_pipeline (OrchestrationPipeline): The processing pipeline for the training set.
        val_pipeline (OrchestrationPipeline): The processing pipeline for the validation set.
        test_pipeline (OrchestrationPipeline): The processing pipeline for the test set.
        verbose (bool): Verbosity for dataset instantiation and loading messages.

    Returns:
        Dict[str, Optional[CachedTensorDataset]]: Datasets for 'train', 'validation', 'test'.
    """
    if verbose:
        print(f"Attempting to load data sources from: {data_sources_json_path}")

    try:
        sources_config_content = load_data_sources_config(data_sources_json_path)
    except Exception as e:
        print(f"Failed to load or parse data sources configuration: {e}")
        return {"train": None, "validation": None, "test": None}

    dataset_info = sources_config_content.get("dataset_configurations", {}).get(dataset_config_name)

    if not dataset_info:
        print(f"Error: Dataset configuration '{dataset_config_name}' not found in {data_sources_json_path}")
        return {"train": None, "validation": None, "test": None}

    base_data_path = dataset_info.get("data_path")
    if not base_data_path:
        print(f"Error: 'data_path' not specified for '{dataset_config_name}' in {data_sources_json_path}")
        return {"train": None, "validation": None, "test": None}

    datasets_out = {}
    pipeline_map = {
        "Train": train_pipeline,
        "Validation": val_pipeline,
        "Test": test_pipeline
    }
    splits_to_load = ["Train", "Validation", "Test"]

    for split_name in splits_to_load:
        current_pipeline = pipeline_map[split_name]
        split_data_dir = os.path.join(base_data_path, split_name)

        if os.path.isdir(split_data_dir):
            if verbose:
                print(f"Initializing {split_name} dataset from: {split_data_dir}")

            datasets_out[split_name.lower()] = CachedTensorDataset(
                cache_dir=split_data_dir,
                transform_pipeline=current_pipeline,
                verbose_dataset=verbose
            )
            if verbose and datasets_out[split_name.lower()]:
                print(
                    f"Successfully initialized {split_name} dataset with {len(datasets_out[split_name.lower()])} items.")
        else:
            if verbose:
                print(f"Warning: {split_name} data directory not found: {split_data_dir}")
            datasets_out[split_name.lower()] = None

    return datasets_out
