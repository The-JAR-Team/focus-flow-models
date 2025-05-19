import torch
import torch.nn as nn
import os
import traceback
from typing import List, Dict, Optional, Union

import onnx
import onnxruntime


def export_to_onnx(
        model: nn.Module,
        dummy_input: torch.Tensor,
        save_path_onnx: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 11,
        device: Optional[Union[str, torch.device]] = None
) -> bool:
    """
    Exports a PyTorch model to ONNX format and verifies the exported model.

    Args:
        model (nn.Module): The trained PyTorch model instance.
        dummy_input (torch.Tensor): An example input tensor with the correct shape
                                    (e.g., batch_size=1, seq_len, num_landmarks, coords).
                                    Used for tracing the model.
        save_path_onnx (str): The file path where the ONNX model will be saved.
        input_names (Optional[List[str]]): Names for the input nodes of the ONNX graph.
                                           Defaults to ['input'].
        output_names (Optional[List[str]]): Names for the output nodes of the ONNX graph.
                                            If the model returns a dict, these should correspond
                                            to the keys. Defaults to ['output'] for single output.
                                            For the multi-task model, this should be
                                            ['regression_scores', 'classification_logits'].
        dynamic_axes (Optional[Dict[str, Dict[int, str]]]): Specifies dynamic axes for inputs/outputs.
                                                            Example: {'input': {0: 'batch_size', 1: 'sequence_length'},
                                                                      'output_name1': {0: 'batch_size'}, ...}
        opset_version (int): The ONNX opset version to use for export.
        device (Optional[Union[str, torch.device]]): The device the model and dummy_input
                                                     should be on for export (e.g., 'cpu', 'cuda').
                                                     If None, uses model's current device or CPU.

    Returns:
        bool: True if export and verification were successful, False otherwise.
    """
    print(f"\n--- Exporting Model to ONNX ({save_path_onnx}) ---")

    # Set defaults if not provided
    if input_names is None:
        input_names = ['input']

    if output_names is None:
        print("Warning: output_names not specified for ONNX export. Defaulting to ['output']. "
              "This might not be suitable for multi-output models.")
        output_names = ['output']

    if device is None:
        try:
            device = next(model.parameters()).device
            print(f"Inferred device for ONNX export: {device}")
        except StopIteration:
            device = torch.device('cpu')
            print(f"Model has no parameters. Using CPU for ONNX export.")
    elif isinstance(device, str):
        device = torch.device(device)

    try:
        output_dir = os.path.dirname(save_path_onnx)
        if output_dir:  # Ensure output_dir is not an empty string if save_path_onnx is just a filename
            os.makedirs(output_dir, exist_ok=True)

        model.eval()
        model.to(device)
        dummy_input = dummy_input.to(device)

        print(f"Exporting with: input_names={input_names}, output_names={output_names}, opset_version={opset_version}")
        if dynamic_axes:
            print(f"Dynamic axes: {dynamic_axes}")

        torch.onnx.export(
            model,
            dummy_input,
            save_path_onnx,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        print("Model successfully exported to ONNX format.")

        print("Verifying ONNX model structure...")
        onnx_model = onnx.load(save_path_onnx)
        onnx.checker.check_model(onnx_model)
        print("ONNX model structure verification successful.")

        print("Testing inference with ONNX Runtime...")
        ort_session = onnxruntime.InferenceSession(save_path_onnx, providers=['CPUExecutionProvider'])
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        print(f"ONNX Runtime inference test successful. Number of outputs: {len(ort_outs)}")
        for i, out_name in enumerate(output_names):
            if i < len(ort_outs):
                print(f"  Output '{out_name}' shape: {ort_outs[i].shape}")
            else:
                print(f"  Warning: More output_names specified than ONNX model outputs received.")
        return True

    except Exception as e:
        print(f"\n!!! ERROR during ONNX export or verification: {e} !!!")
        traceback.print_exc()
        if os.path.exists(save_path_onnx):
            try:
                os.remove(save_path_onnx)
                print(f"Removed potentially corrupted file: {save_path_onnx}")
            except OSError as ose:
                print(f"Warning: Could not remove potentially corrupted file {save_path_onnx}: {ose}")
        return False
