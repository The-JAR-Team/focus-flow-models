import os
import torch
import torch.nn as nn
# import torch.nn.functional as F # Not used in this snippet
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Dict, Any, Tuple, Optional, List

# Assuming these imports are correct relative to your project structure
from Model_Training.utils.onnx_exporter import export_to_onnx
from Model_Training.utils.plotting import plot_hf_training_history


class PlottingCallback(TrainerCallback):
    """
    A TrainerCallback that generates and saves plots of training and evaluation
    metrics at the end of training, including confusion matrices.

    MODIFIED:
    - Constructor now accepts `idx_to_name_map`, `confusion_matrix_eval_filename`,
      and `confusion_matrix_test_filename`.
    - Passes these parameters to `plot_hf_training_history`.
    """

    def __init__(self,
                 plots_save_dir: Optional[str] = None,
                 idx_to_name_map: Optional[Dict[int, str]] = None,  # Added for CM labels
                 loss_plot_filename: str = "training_validation_loss.png",
                 lr_plot_filename: str = "learning_rate.png",
                 regression_metrics_plot_filename: str = "regression_metrics.png",
                 classification_metrics_plot_filename: str = "classification_metrics.png",
                 confusion_matrix_eval_filename: str = "confusion_matrix_eval.png",  # Added
                 confusion_matrix_test_filename: str = "confusion_matrix_test.png"  # Added
                 ):
        super().__init__()
        self.plots_save_dir_config = plots_save_dir
        self.idx_to_name_map = idx_to_name_map  # Store this for passing to plotting function
        self.loss_plot_filename = loss_plot_filename
        self.lr_plot_filename = lr_plot_filename
        self.regression_metrics_plot_filename = regression_metrics_plot_filename
        self.classification_metrics_plot_filename = classification_metrics_plot_filename
        self.confusion_matrix_eval_filename = confusion_matrix_eval_filename  # Store this
        self.confusion_matrix_test_filename = confusion_matrix_test_filename  # Store this
        print("PlottingCallback initialized (with confusion matrix support).")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the end of training to generate and save plots.
        """
        print("\nPlottingCallback: Training ended. Generating plots...")
        if state.log_history:
            # Determine the actual directory to save plots
            if self.plots_save_dir_config:  # If a specific dir was passed during init
                actual_plots_dir = self.plots_save_dir_config
            else:  # Fallback to the trainer's main output directory
                actual_plots_dir = args.output_dir

            os.makedirs(actual_plots_dir, exist_ok=True)  # Ensure the directory exists

            # Call the main plotting function with all necessary parameters
            plot_hf_training_history(
                log_history=state.log_history,
                output_dir=actual_plots_dir,
                idx_to_name_map=self.idx_to_name_map,  # Pass the map for CM labels
                loss_plot_filename=self.loss_plot_filename,
                lr_plot_filename=self.lr_plot_filename,
                regression_metrics_plot_filename=self.regression_metrics_plot_filename,
                classification_metrics_plot_filename=self.classification_metrics_plot_filename,
                confusion_matrix_eval_filename=self.confusion_matrix_eval_filename,  # Pass CM filename
                confusion_matrix_test_filename=self.confusion_matrix_test_filename  # Pass CM filename
            )
            print(f"Plots (including confusion matrices if data available) saved in {actual_plots_dir}")
        else:
            print("PlottingCallback: No log history found in TrainerState. Skipping plot generation.")


class OnnxExportCallback(TrainerCallback):  # No changes to this class
    """
    A TrainerCallback that exports the trained model to ONNX format
    at the end of successful training.
    """

    def __init__(self,
                 onnx_model_full_save_path: str,
                 opset_version: int = 11,
                 input_shape_for_onnx: Tuple[int, int, int] = (30, 478, 3),
                 onnx_input_names: Optional[List[str]] = None,
                 onnx_output_names: Optional[List[str]] = None,
                 onnx_dynamic_axes: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.onnx_model_full_save_path = onnx_model_full_save_path
        self.opset_version = opset_version
        self.input_shape_for_onnx = input_shape_for_onnx

        self.onnx_input_names = onnx_input_names if onnx_input_names is not None else ['input']
        self.onnx_output_names = onnx_output_names if onnx_output_names is not None else ['regression_scores',
                                                                                          'classification_logits']

        if onnx_dynamic_axes is None:
            self.onnx_dynamic_axes = {self.onnx_input_names[0]: {0: 'batch_size', 1: 'sequence_length'}}
            for name in self.onnx_output_names:
                self.onnx_dynamic_axes[name] = {0: 'batch_size'}
        else:
            self.onnx_dynamic_axes = onnx_dynamic_axes

        # Get the filename for the print statement
        model_filename_for_print = os.path.basename(self.onnx_model_full_save_path)
        print(f"OnnxExportCallback initialized. Model will be saved as {model_filename_for_print}.")
        # print(f"  ONNX Input Names: {self.onnx_input_names}") # Already in original
        # print(f"  ONNX Output Names: {self.onnx_output_names}") # Already in original
        # print(f"  ONNX Dynamic Axes: {self.onnx_dynamic_axes}") # Already in original

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                     model: nn.Module = None, **kwargs):
        print("\nOnnxExportCallback: Training ended.")
        if model is None:
            print("OnnxExportCallback: No model provided to on_train_end. Skipping ONNX export.")
            return

        if not state.is_world_process_zero:  # Ensure export only on main process
            print("OnnxExportCallback: Not on main process. Skipping ONNX export.")
            return

        onnx_save_path = self.onnx_model_full_save_path
        # Batch size of 1 for the dummy input for ONNX tracing
        dummy_input = torch.randn(1, *self.input_shape_for_onnx, device='cpu')  # Create on CPU

        print(f"OnnxExportCallback: Attempting to export model to {onnx_save_path}")
        # print(f"  Dummy input shape for ONNX export: {dummy_input.shape}") # Already in original

        export_success = export_to_onnx(
            model=model,
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
