import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Dict, Any, Tuple, Optional, List

from Model_Training.utils.onnx_exporter import export_to_onnx
from Model_Training.utils.plotting import plot_hf_training_history


class PlottingCallback(TrainerCallback):
    """
    A TrainerCallback that generates and saves plots of training and evaluation
    metrics at the end of training.
    """

    def __init__(self,
                 plots_save_dir: Optional[str] = None,
                 loss_plot_filename: str = "training_validation_loss.png",
                 lr_plot_filename: str = "learning_rate.png",
                 regression_metrics_plot_filename: str = "regression_metrics.png",
                 classification_metrics_plot_filename: str = "classification_metrics.png"):
        super().__init__()
        self.plots_save_dir_config = plots_save_dir
        self.loss_plot_filename = loss_plot_filename
        self.lr_plot_filename = lr_plot_filename
        self.regression_metrics_plot_filename = regression_metrics_plot_filename
        self.classification_metrics_plot_filename = classification_metrics_plot_filename
        print("PlottingCallback initialized.")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the end of training.
        """
        print("\nPlottingCallback: Training ended. Generating plots...")
        if state.log_history:
            if self.plots_save_dir_config:
                actual_plots_dir = self.plots_save_dir_config
            else:
                actual_plots_dir = args.output_dir  # Fallback to trainer's output dir

            os.makedirs(actual_plots_dir, exist_ok=True)  # Ensure it exists

            plot_hf_training_history(
                log_history=state.log_history,
                output_dir=actual_plots_dir,
                loss_plot_filename=self.loss_plot_filename,
                lr_plot_filename=self.lr_plot_filename,
                regression_metrics_plot_filename=self.regression_metrics_plot_filename,
                classification_metrics_plot_filename=self.classification_metrics_plot_filename
            )
            print(f"Plots saved in {actual_plots_dir}")
        else:
            print("PlottingCallback: No log history found. Skipping plot generation.")


class OnnxExportCallback(TrainerCallback):
    """
    A TrainerCallback that exports the trained model to ONNX format
    at the end of successful training.
    """

    def __init__(self,
                 onnx_model_full_save_path: str,
                 onnx_model_filename: str = "model.onnx",
                 opset_version: int = 11,
                 input_shape_for_onnx: Tuple[int, int, int] = (30, 478, 3),  # (seq_len, num_landmarks, num_coords)
                 onnx_input_names: Optional[List[str]] = None,  # Default ['input'] will be used by export_to_onnx
                 onnx_output_names: Optional[List[str]] = None,  # e.g. ['regression_scores', 'classification_logits']
                 onnx_dynamic_axes: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.onnx_model_full_save_path = onnx_model_full_save_path
        self.onnx_model_filename = onnx_model_filename
        self.opset_version = opset_version
        self.input_shape_for_onnx = input_shape_for_onnx  # (seq_len, num_landmarks, coords)

        # Set defaults if not provided, matching export_to_onnx defaults or typical use
        self.onnx_input_names = onnx_input_names if onnx_input_names is not None else ['input']

        # For the multi-task model, these are crucial.
        # The user should configure these in experiment_config.py and pass them when creating this callback.
        self.onnx_output_names = onnx_output_names if onnx_output_names is not None else ['regression_scores',
                                                                                          'classification_logits']

        # Default dynamic axes if none are provided
        if onnx_dynamic_axes is None:
            self.onnx_dynamic_axes = {self.onnx_input_names[0]: {0: 'batch_size', 1: 'sequence_length'}}
            for name in self.onnx_output_names:
                self.onnx_dynamic_axes[name] = {0: 'batch_size'}
        else:
            self.onnx_dynamic_axes = onnx_dynamic_axes

        print(f"OnnxExportCallback initialized. Model will be saved as {self.onnx_model_filename}.")
        print(f"  ONNX Input Names: {self.onnx_input_names}")
        print(f"  ONNX Output Names: {self.onnx_output_names}")
        print(f"  ONNX Dynamic Axes: {self.onnx_dynamic_axes}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                     model: nn.Module = None, **kwargs):
        """
        Called at the end of training. Exports the model if available.
        The `model` kwarg should be the trained model instance.
        """
        print("\nOnnxExportCallback: Training ended.")
        if model is None:
            print("OnnxExportCallback: No model provided to on_train_end. Skipping ONNX export.")
            return

        if not state.is_world_process_zero:
            # Ensure export only happens on the main process in distributed training
            print("OnnxExportCallback: Not on main process. Skipping ONNX export.")
            return

        # Construct the full save path for the ONNX model
        onnx_save_path = self.onnx_model_full_save_path

        # Create a dummy input tensor for tracing
        # Batch size of 1 for the dummy input
        # Shape: (1, seq_len, num_landmarks, num_coords)
        dummy_input = torch.randn(1, *self.input_shape_for_onnx, device='cpu')  # Create on CPU first

        print(f"OnnxExportCallback: Attempting to export model to {onnx_save_path}")
        print(f"  Dummy input shape for ONNX export: {dummy_input.shape}")

        export_success = export_to_onnx(
            model=model,  # The trained model passed by the Trainer
            dummy_input=dummy_input,
            save_path_onnx=onnx_save_path,
            input_names=self.onnx_input_names,
            output_names=self.onnx_output_names,
            dynamic_axes=self.onnx_dynamic_axes,
            opset_version=self.opset_version,
            device='cpu'  # Perform export on CPU; model will be moved by export_to_onnx
        )

        if export_success:
            print(f"OnnxExportCallback: Model successfully exported to {onnx_save_path}")
        else:
            print(f"OnnxExportCallback: Failed to export model to ONNX.")