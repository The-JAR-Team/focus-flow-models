# engagement_hf_trainer/pipelines/orchestration.py
from typing import List, Tuple, Dict, Any
import torch
from Model_Training.pipelines.base_stage import BaseStage


class OrchestrationPipeline:
    def __init__(self, stages: List[BaseStage]):
        self.stages = stages if stages is not None else []

    def run(self, x_initial: torch.Tensor, y_initial: Dict[str, torch.Tensor], verbose_pipeline: bool = False) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        x_current, y_current = x_initial, y_initial
        if verbose_pipeline:
            print(f"Starting OrchestrationPipeline with {len(self.stages)} stages.")

        for i, stage in enumerate(self.stages):
            if verbose_pipeline:
                print(f"  Pipeline: Running stage {i + 1}/{len(self.stages)}: {stage.__class__.__name__}")
            try:
                # Individual stages can have their own verbosity, controlled during their instantiation
                x_current, y_current = stage(x_current, y_current)
            except Exception as e:
                print(f"Error in pipeline stage {stage.__class__.__name__}: {e}")
                # Decide on error handling: re-raise, skip stage, or return original data
                # For now, let's re-raise or log and continue with current data
                # For robustness, you might want to return x_initial, y_initial or x_current, y_current
                raise  # Or handle more gracefully

        if verbose_pipeline:
            print("OrchestrationPipeline finished.")
        return x_current, y_current

    def __call__(self, x_initial: torch.Tensor, y_initial: Dict[str, torch.Tensor], verbose_pipeline: bool = False) -> \
    Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.run(x_initial, y_initial, verbose_pipeline)