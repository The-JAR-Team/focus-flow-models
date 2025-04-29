import torch
import torch.nn as nn
import onnx
import onnxruntime
import os
import traceback
from Model.engagement_regression_model import EngagementRegressionModel


# ================================================
# === ONNX Export Function ===
# ================================================
def export_to_onnx(
    model: EngagementRegressionModel,
    dummy_input: torch.Tensor,
    save_path_onnx: str,
    device: torch.device,
    opset_version: int = 11
    ) -> bool:
    """
    Exports the PyTorch model to ONNX format and verifies the exported model.

    Args:
        model (EngagementRegressionModel): The trained PyTorch model instance.
        dummy_input (torch.Tensor): An example input tensor with the correct shape
                                    (batch_size=1, seq_len, num_landmarks, coords).
                                    Used for tracing the model.
        save_path_onnx (str): The file path where the ONNX model will be saved.
        device (torch.device): The device the model and dummy_input should be on for export.
        opset_version (int): The ONNX opset version to use for export.

    Returns:
        bool: True if export and verification were successful, False otherwise.
    """
    print(f"\n--- Exporting Model to ONNX ({save_path_onnx}) ---")
    try:
        # Ensure model is in evaluation mode and on the correct device
        model.eval()
        model.to(device)
        # Ensure dummy input is also on the correct device
        dummy_input = dummy_input.to(device)

        # Perform the export
        torch.onnx.export(
            model,                     # The model to export
            dummy_input,               # Model input (used for tracing)
            save_path_onnx,            # Where to save the model
            export_params=True,        # Store the trained parameter weights inside the model file
            opset_version=opset_version,  # ONNX version to export the model to
            do_constant_folding=True,  # Whether to execute constant folding for optimization
            input_names = ['input'],   # The model's input names
            output_names = ['output'], # The model's output names
            dynamic_axes={             # Specify variable length axes
                'input' : {0 : 'batch_size', 1: 'sequence_length'}, # Batch size and seq length can vary
                'output' : {0 : 'batch_size'}  # Output batch size varies with input
            }
        )
        print("Model successfully exported to ONNX format.")

        # --- Verification ---
        print("Verifying ONNX model...")
        # Load the exported model
        onnx_model = onnx.load(save_path_onnx)
        # Check if the model is structurally valid
        onnx.checker.check_model(onnx_model)
        print("ONNX model structure verification successful.")

        # Optional: Test inference with ONNX Runtime for numerical consistency check
        print("Testing inference with ONNX Runtime...")
        ort_session = onnxruntime.InferenceSession(save_path_onnx, providers=['CPUExecutionProvider']) # Use CPU provider for broader compatibility
        # Prepare input for ONNX Runtime (needs to be numpy array)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        # Run inference
        ort_outs = ort_session.run(None, ort_inputs)
        print("ONNX Runtime inference test successful.")
        # Compare PyTorch output with ONNX output (optional, requires running PyTorch model again)
        # with torch.no_grad():
        #     pytorch_out = model(dummy_input)
        # np.testing.assert_allclose(pytorch_out.cpu().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
        # print("PyTorch and ONNX Runtime outputs match numerically.")

        return True

    except ImportError:
        print("\n!!! ERROR: onnx or onnxruntime package not found. Cannot export to ONNX. !!!")
        print("   Install using: pip install onnx onnxruntime")
        return False
    except Exception as e:
        print(f"\n!!! ERROR during ONNX export or verification: {e} !!!")
        traceback.print_exc()
        # Clean up potentially corrupted ONNX file if export failed mid-way
        if os.path.exists(save_path_onnx):
            try:
                os.remove(save_path_onnx)
                print(f"Removed potentially corrupted file: {save_path_onnx}")
            except OSError:
                print(f"Warning: Could not remove potentially corrupted file: {save_path_onnx}")
        return False
