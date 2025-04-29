import torch
import torch.nn as nn


# ================================================
# === Model Definition ===
# ================================================
class EngagementRegressionModel(nn.Module):
    """ GRU-based model for engagement regression. """
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.5, bidirectional=True):
        """
        Initializes the EngagementRegressionModel.

        Args:
            input_dim (int): The number of expected features in the input x.
            hidden_dim (int): The number of features in the hidden state h.
            output_dim (int): The number of output features (1 for regression).
            num_layers (int): Number of recurrent layers.
            dropout (float): If non-zero, introduces a Dropout layer on the outputs of each
                             GRU layer except the last layer, with dropout probability equal to dropout.
            bidirectional (bool): If True, becomes a bidirectional GRU.
        """
        super().__init__()
        self.input_dim = input_dim
        self.bidirectional = bidirectional

        # Layer normalization for input features at each time step
        self.frame_norm = nn.LayerNorm(input_dim)

        # GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)

        # Calculate the output dimension of the GRU layer
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers
        self.fc1 = nn.Linear(gru_output_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        # Sigmoid activation to constrain output between 0 and 1
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, num_landmarks, coords).

        Returns:
            torch.Tensor: Output tensor of shape (batch, output_dim).
        """
        # Input shape: (batch, seq_len, num_landmarks, coords)
        batch_size, seq_len, num_landmarks, coords = x.shape
        # Reshape to combine landmarks and coords: (batch, seq_len, num_landmarks * coords)
        x = x.reshape(batch_size, seq_len, -1)

        # Validate input dimension after reshaping
        if x.shape[2] != self.input_dim:
             raise ValueError(f"Input dim mismatch: Expected {self.input_dim}, Got {x.shape[2]}")

        # Apply layer normalization
        x = self.frame_norm(x)

        # Pass through GRU
        # gru_out shape: (batch, seq_len, hidden_dim * num_directions)
        # hn shape: (num_layers * num_directions, batch, hidden_dim)
        gru_out, hn = self.gru(x)

        # Extract the hidden state of the last time step
        if self.bidirectional:
            # Concatenate the last hidden states from forward (hn[-2]) and backward (hn[-1]) directions
            # Shape becomes (batch, hidden_dim * 2)
            last_hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        else:
            # Use the last hidden state from the single direction
            # Shape becomes (batch, hidden_dim)
            last_hidden = hn[-1,:,:]

        # Apply dropout
        out = self.dropout(last_hidden)
        # Pass through fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        # Apply final activation
        out = self.output_activation(out) # Ensure output is in [0, 1]

        # Output shape: (batch, output_dim) which is (batch, 1) for regression
        return out

if __name__ == '__main__':
    # Example usage: Instantiate the model and print summary
    print("--- Model Definition Example ---")
    # Use example dimensions (replace with actual values from config if needed)
    example_input_dim = 478 * 3
    example_hidden_dim = 256
    model = EngagementRegressionModel(input_dim=example_input_dim, hidden_dim=example_hidden_dim)
    print(model)

    # Example forward pass with dummy data
    batch_s, seq_l, landmarks, coords_ = 4, 30, 478, 3
    dummy_input = torch.randn(batch_s, seq_l, landmarks, coords_)
    try:
        output = model(dummy_input)
        print(f"\nDummy input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error during dummy forward pass: {e}")

