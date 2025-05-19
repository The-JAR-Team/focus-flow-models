# engagement_hf_trainer/pipelines/stages/base_stage.py
from typing import Tuple, Dict, Any
import torch

class BaseStage:
    def __init__(self, verbose: bool = False, **kwargs):
        self.verbose = verbose
        # Store any other common kwargs

    def process(self, x: torch.Tensor, y: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Processes the input tensor (x) and label dictionary (y).
        Stages should override this method.
        If a stage only modifies x, it should return (processed_x, y).
        If a stage only modifies y, it should return (x, processed_y).
        """
        if self.verbose:
            print(f"Running {self.__class__.__name__}")
        return x, y # Default: pass through

    def __call__(self, x: torch.Tensor, y: Dict[str, torch.Tensor], **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # kwargs can be used to pass additional dynamic info to a stage if needed
        return self.process(x, y)