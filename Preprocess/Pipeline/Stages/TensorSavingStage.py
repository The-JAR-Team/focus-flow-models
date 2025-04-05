import os
import torch
from Preprocess.Pipeline.PipelineStage import PipelineStage
from Preprocess.Pipeline.Encapsulation.TensorStackingResult import TensorStackingResult

class TensorSavingStage(PipelineStage):
    """
    A pipeline stage that saves the tensor stacking result to disk and/or loads it from cache.
    It checks for an existing file under:
       {cache_root}/PipelineResult/{pipeline_version}/{dataset_type}/{subfolder}/{clip_folder}_{pipeline_version}.pt
    If found, it loads and returns the result. Otherwise, it saves the provided result.
    """
    def __init__(self, pipeline_version: str, cache_root: str):
        """
        Parameters:
          pipeline_version: str (e.g., "01")
          cache_root: str, root directory for cached pipeline results.
        """
        self.pipeline_version = pipeline_version
        self.cache_root = cache_root

    def process(self, tensor_stacking_result: TensorStackingResult, verbose=True):
        """
        Parameters:
          tensor_stacking_result (TensorStackingResult): The result from the previous stage,
              containing 'tensor_stack', 'label', 'clip_folder', and 'dataset_type'.
          verbose (bool): If True, prints detailed status messages.

        Returns:
          TensorStackingResult: The loaded or saved tensor stacking result.
        """
        # Retrieve clip_folder and dataset_type from the result.
        clip_folder = tensor_stacking_result.clip_folder
        dataset_type = tensor_stacking_result.dataset_type  # Must be set upstream.
        # Use the first 6 characters of clip_folder to create a subdirectory.
        subfolder = clip_folder[:6]

        # Build the save directory:
        # {cache_root}/PipelineResult/{pipeline_version}/{dataset_type}/{subfolder}/
        save_dir = os.path.join(self.cache_root, "PipelineResult", self.pipeline_version, dataset_type, subfolder)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{clip_folder}_{self.pipeline_version}.pt")

        if os.path.exists(save_path):
            if verbose:
                print("-------")
                print("TensorSaving stage")
                print(f"Clip folder: {clip_folder}")
                print(f"Cached tensor result found for clip {clip_folder} at {save_path}")
                print("passed!")
                print("-------")
            data = torch.load(save_path)
            return TensorStackingResult(tensor_stack=data["tensor_stack"],
                                         label=data["label"],
                                         clip_folder=clip_folder,
                                         dataset_type=dataset_type)
        else:
            torch.save({"tensor_stack": tensor_stacking_result.tensor_stack,
                        "label": tensor_stacking_result.label},
                       save_path)
            if verbose:
                print("-------")
                print("TensorSaving stage")
                print(f"Clip folder: {clip_folder}")
                print(f"Saved tensor result for clip {clip_folder} at {save_path}")
                print("passed!")
                print("-------")
            return tensor_stacking_result


# Example usage (for testing independently)
if __name__ == "__main__":
    import torch
    from Preprocess.Pipeline.Encapsulation.TensorStackingResult import TensorStackingResult
    # Dummy tensor stacking result for testing:
    dummy_tensor = torch.randn(100, 478, 3)
    dummy_label = {"engagement": 2}
    # Set dummy dataset_type, for example, "Train"
    dummy_result = TensorStackingResult(tensor_stack=dummy_tensor, label=dummy_label, clip_folder="1100011002", dataset_type="Train")

    # Replace with your actual CACHE_DIR from DaiseeConfig.py.
    cache_root = r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\dataset\DaiseeData\Cache"
    stage = TensorSavingStage(pipeline_version="01", cache_root=cache_root)
    output = stage.process(dummy_result, verbose=True)
    print("Output tensor stack shape:", output.tensor_stack.shape)
