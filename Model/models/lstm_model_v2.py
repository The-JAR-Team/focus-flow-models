# Model/models/lstm_model_v2.py

import torch
import torch.nn as nn

# --- Model V2 (Simplified) Hyperparameters ---
INPUT_DIM_V2 = 478 * 3
HIDDEN_DIM_V2 = 256  # Reduced hidden dim
NUM_LSTM_LAYERS_V2 = 2  # Reduced layers
DROPOUT_RATE_V2 = 0.4

# ================================================
# === Model Definition: LstmModelV2 (Simplified) ===
# ================================================
class LstmModelV2(nn.Module): # Keep class name V2
    """ LSTM-based model V2 for engagement regression (Simplified). """
    def __init__(self, input_dim=INPUT_DIM_V2, hidden_dim=HIDDEN_DIM_V2, output_dim=1,
                 num_layers=NUM_LSTM_LAYERS_V2, dropout=DROPOUT_RATE_V2, bidirectional=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.frame_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional
        )
        lstm_output_dim = hidden_dim * self.num_directions

        self.head_dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim // 2)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_landmarks, coords = x.shape
        x = x.view(batch_size, seq_len, -1)
        if x.shape[2] != self.input_dim:
             raise ValueError(f"Input dim mismatch: Expected {self.input_dim}, Got {x.shape[2]}")

        x = self.frame_norm(x)
        lstm_out, (hn, cn) = self.lstm(x)

        if self.bidirectional:
            last_hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        else:
            last_hidden = hn[-1,:,:]

        out = self.head_dropout(last_hidden)
        out = self.fc1(out)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.output_activation(out)
        return out

if __name__ == '__main__':
    print("--- Model Definition Example (LstmModelV2 Simplified) ---")
    model_v2 = LstmModelV2()
    print(model_v2)
    num_params = sum(p.numel() for p in model_v2.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {num_params:,}")

    batch_s, seq_l, landmarks, coords_ = 4, 30, 478, 3
    dummy_input = torch.randn(batch_s, seq_l, landmarks, coords_)
    print(f"\nDummy input shape: {dummy_input.shape}")
    try:
        output = model_v2(dummy_input)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error during dummy forward pass: {e}")