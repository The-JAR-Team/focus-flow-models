# Model/engagement_regression_model.py
import torch
import torch.nn as nn
from typing import List, Optional


# ================================================
# === Configurable Model Definition ===
# ================================================
class ConfigurableEngagementModel(nn.Module):
    """
    A sequence model that uses a pre-instantiated list of layers
    provided from the configuration.

    Handles input reshaping, initial normalization, and special processing
    for GRU/LSTM layers found within the provided list.
    """

    def __init__(self, input_dim: int, model_layers: List[nn.Module]):
        """
        Initializes the ConfigurableEngagementModel.

        Args:
            input_dim (int): The number of expected features in the input x *after*
                             reshaping (num_landmarks * coords).
            model_layers (List[nn.Module]): A list containing *instantiated* nn.Module
                                             objects (e.g., [nn.GRU(...), nn.Linear(...)]).
        """
        super().__init__()
        self.input_dim = input_dim

        # --- Standard Input Handling ---
        # Layer normalization applied to input features at each time step
        self.initial_frame_norm = nn.LayerNorm(input_dim)
        # ---

        # --- Store and Register Provided Layers ---
        # Use nn.ModuleList to ensure layers are correctly registered
        # (parameters are found by optimizer, state_dict works, etc.)
        self.layers = nn.ModuleList(model_layers)
        # ---

        # --- Identify GRU/LSTM for forward pass logic ---
        self.recurrent_layer_index = -1
        self.is_recurrent_bidirectional = False
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (nn.GRU, nn.LSTM)):
                if self.recurrent_layer_index != -1:
                    # Simple setup limitation: only handle one RNN layer automatically
                    print(f"Warning: Multiple recurrent layers found. Special forward pass logic "
                          f"will only apply to the first one found at index {self.recurrent_layer_index}.")
                else:
                    self.recurrent_layer_index = i
                    # Check the bidirectional attribute directly on the instance
                    self.is_recurrent_bidirectional = getattr(layer, 'bidirectional', False)
                    print(
                        f"Identified recurrent layer {layer.__class__.__name__} at index {i}. Bidirectional: {self.is_recurrent_bidirectional}")
                    # No need to break, just identify the first one for now

        if self.recurrent_layer_index == -1:
            print("Warning: No GRU or LSTM layer found in the provided list. "
                  "Model will process sequentially without special hidden state extraction.")

        print("--- Configurable Model Initialized with Provided Layers ---")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass using the provided layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, num_landmarks, coords).

        Returns:
            torch.Tensor: Output tensor from the final layer in the list.
        """
        # --- Standard Input Handling ---
        batch_size, seq_len, num_landmarks, coords = x.shape
        x = x.reshape(batch_size, seq_len, -1)  # Reshape
        if x.shape[2] != self.input_dim:
            raise ValueError(f"Input dim mismatch during forward: Expected {self.input_dim}, Got {x.shape[2]}")
        x = self.initial_frame_norm(x)  # Apply initial normalization
        # ---

        # --- Process through provided layers ---
        for i, layer in enumerate(self.layers):
            if i == self.recurrent_layer_index:
                # --- Special handling for the identified GRU/LSTM layer ---
                if isinstance(layer, nn.GRU):
                    rnn_output, hn = layer(x)
                elif isinstance(layer, nn.LSTM):
                    rnn_output, (hn, cn) = layer(x)
                else:
                    # Should not happen if index identification is correct
                    raise TypeError(f"Layer at recurrent_layer_index {i} is not GRU or LSTM!")

                # Extract the final hidden state based on bidirectionality
                if self.is_recurrent_bidirectional:
                    # Concatenate final states of forward and backward layers
                    # hn shape: (num_layers * num_directions, batch, hidden_size)
                    # Forward is hn[-2], Backward is hn[-1]
                    x = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
                else:
                    # Use final state of the single direction layer
                    # hn shape: (num_layers, batch, hidden_size)
                    x = hn[-1, :, :]
                # The output 'x' is now the final hidden state(s), shape (batch, gru_output_dim)
                # This output is passed to the *next* layer in the list.

            else:
                # --- Apply all other layers sequentially ---
                # Layers after the GRU/LSTM will operate on the hidden state output.
                # Layers before the GRU/LSTM operate on the sequence.
                # Need to handle potential dimension issues for layers like BatchNorm1d
                if isinstance(layer, nn.BatchNorm1d) and x.dim() == 3:
                    # BatchNorm1d expects (N, C) or (N, C, L)
                    # If input is (batch, seq, features), transpose
                    x = x.permute(0, 2, 1)  # -> (batch, features, seq)
                    x = layer(x)
                    x = x.permute(0, 2, 1)  # -> (batch, seq, features) - Transpose back
                elif isinstance(layer, nn.BatchNorm1d) and x.dim() == 2:
                    # If input is (batch, features) (e.g., after GRU hidden state)
                    x = layer(x)  # Apply directly
                elif isinstance(layer, nn.LayerNorm) and x.dim() == 3:
                    # Applied to the sequence
                    x = layer(x)
                elif isinstance(layer, nn.LayerNorm) and x.dim() == 2:
                    # Applied to hidden state output
                    x = layer(x)
                else:
                    # Apply other layers (Linear, Activation, Dropout, etc.)
                    x = layer(x)

        # The final value of 'x' after the loop is the output of the last layer
        return x


# Example usage (can be run if this file is executed directly)
if __name__ == '__main__':
    print("--- Configurable Model Definition Example ---")
    # Define example parameters needed for layer instantiation
    example_input_dim = 50
    example_hidden_dim = 32
    example_output_dim = 1
    example_num_layers = 1
    example_bidirectional = False

    # Create an example list of instantiated layers
    example_layers = [
        nn.GRU(example_input_dim, example_hidden_dim, num_layers=example_num_layers,
               batch_first=True, bidirectional=example_bidirectional),
        nn.Linear(example_hidden_dim * (2 if example_bidirectional else 1), example_output_dim),
        nn.Sigmoid()
    ]

    try:
        model = ConfigurableEngagementModel(
            input_dim=example_input_dim,
            model_layers=example_layers
        )
        print("\nModel Structure (Uses Provided Layers):")
        print(model)  # Note: This print shows the ModuleList containing the layers

        # Example forward pass with dummy data
        batch_s, seq_l, landmarks, coords_ = 4, 10, 5, 10  # Match example_input_dim = 50
        dummy_input = torch.randn(batch_s, seq_l, landmarks, coords_)
        output = model(dummy_input)
        print(f"\nDummy input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        # Verify output shape matches expected output_dim
        assert output.shape == (batch_s, example_output_dim)
        print("Forward pass successful and output shape matches.")

    except Exception as e:
        print(f"\nAn error occurred during example usage: {e}")
        import traceback

        traceback.print_exc()
