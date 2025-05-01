# Model/models/gru_attention_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Model Hyperparameters ---
INPUT_DIM_GA = 478 * 3
HIDDEN_DIM_GA = 256  # Similar to original GRU / simplified LSTM
NUM_GRU_LAYERS_GA = 2
DROPOUT_RATE_GA = 0.4

# ================================================
# === Model Definition: GruAttentionModel ===
# ================================================
class GruAttentionModel(nn.Module):
    """ GRU-based model with simple attention mechanism. """
    def __init__(self, input_dim=INPUT_DIM_GA, hidden_dim=HIDDEN_DIM_GA, output_dim=1,
                 num_layers=NUM_GRU_LAYERS_GA, dropout=DROPOUT_RATE_GA, bidirectional=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.gru_output_dim = hidden_dim * self.num_directions

        self.frame_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional
        )

        # --- Simple Attention Mechanism ---
        # Layer to compute attention scores from GRU outputs
        self.attention_layer = nn.Linear(self.gru_output_dim, 1)

        # --- MLP Head ---
        self.head_dropout = nn.Dropout(dropout)
        # Input to head is the context vector size == gru_output_dim
        self.fc1 = nn.Linear(self.gru_output_dim, hidden_dim // 2)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_landmarks, coords = x.shape
        x = x.view(batch_size, seq_len, -1)
        if x.shape[2] != self.input_dim:
             raise ValueError(f"Input dim mismatch: Expected {self.input_dim}, Got {x.shape[2]}")

        x = self.frame_norm(x)
        # gru_out shape: (batch, seq_len, gru_output_dim)
        # hn shape: (num_layers * num_directions, batch, hidden_dim)
        gru_out, hn = self.gru(x)

        # --- Attention Calculation ---
        # Calculate attention scores for each time step
        # attn_scores shape: (batch, seq_len, 1)
        attn_scores = self.attention_layer(gru_out)
        # Apply tanh (optional, common practice)
        attn_scores = torch.tanh(attn_scores)

        # Normalize scores to get weights using softmax
        # attn_weights shape: (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1) # Normalize across the sequence length dimension

        # Calculate context vector (weighted sum of GRU outputs)
        # Use einsum for clarity or basic matmul + sum
        # context_vector shape: (batch, gru_output_dim)
        # Element-wise multiply gru_out by weights and sum across seq_len
        context_vector = torch.sum(gru_out * attn_weights, dim=1)

        # --- MLP Head Processing ---
        out = self.head_dropout(context_vector)
        out = self.fc1(out)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.output_activation(out)

        return out

if __name__ == '__main__':
    print("--- Model Definition Example (GruAttentionModel) ---")
    model_ga = GruAttentionModel()
    print(model_ga)
    num_params = sum(p.numel() for p in model_ga.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {num_params:,}")

    batch_s, seq_l, landmarks, coords_ = 4, 30, 478, 3
    dummy_input = torch.randn(batch_s, seq_l, landmarks, coords_)
    print(f"\nDummy input shape: {dummy_input.shape}")
    try:
        output = model_ga(dummy_input)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error during dummy forward pass: {e}")