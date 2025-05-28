import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union


class EngagementMultiTaskGRUAttentionModel(nn.Module):
    """
    GRU-based model with attention, featuring separate heads for regression and
    classification tasks. Designed for Hugging Face Trainer.
    Loss functions are passed in during initialization.
    """

    def __init__(self,
                 input_dim: int,
                 regression_loss_fn: nn.Module,  # Pass instantiated loss function
                 classification_loss_fn: nn.Module,  # Pass instantiated loss function
                 hidden_dim: int = 256,
                 num_gru_layers: int = 2,
                 dropout_rate: float = 0.4,
                 bidirectional_gru: bool = True,
                 regression_output_dim: int = 1,
                 num_classes: int = 5,
                 regression_loss_weight: float = 1.0,
                 classification_loss_weight: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        self.bidirectional_gru = bidirectional_gru
        self.num_directions = 2 if bidirectional_gru else 1
        self.gru_output_dim = hidden_dim * self.num_directions

        self.regression_output_dim = regression_output_dim
        self.num_classes = num_classes
        self.regression_loss_weight = regression_loss_weight
        self.classification_loss_weight = classification_loss_weight

        # Store the provided loss functions
        self.regression_loss_fn = regression_loss_fn
        self.classification_loss_fn = classification_loss_fn

        # Shared Body
        self.frame_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_gru_layers, batch_first=True,
            dropout=dropout_rate if num_gru_layers > 1 else 0,
            bidirectional=bidirectional_gru
        )
        self.attention_layer = nn.Linear(self.gru_output_dim, 1)
        self.shared_dropout = nn.Dropout(dropout_rate)

        # Regression Head
        self.reg_fc1 = nn.Linear(self.gru_output_dim, hidden_dim // 2)
        self.reg_relu = nn.ReLU()
        self.reg_fc2 = nn.Linear(hidden_dim // 2, regression_output_dim)
        self.reg_output_activation = nn.Sigmoid()

        # Classification Head
        self.cls_fc1 = nn.Linear(self.gru_output_dim, hidden_dim // 2)
        self.cls_relu = nn.ReLU()
        self.cls_fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self,
                x: torch.Tensor,
                labels: Optional[Dict[str, torch.Tensor]] = None
                ) -> Dict[str, torch.Tensor]:
        # ... (rest of the forward pass for calculating context_vector_dropped is the same)
        batch_size, seq_len, num_landmarks, coords = x.shape
        x_reshaped = x.view(batch_size, seq_len, -1)

        if x_reshaped.shape[2] != self.input_dim:
            raise ValueError(f"Input feature dimension mismatch. Expected {self.input_dim}, got {x_reshaped.shape[2]}.")

        normalized_x = self.frame_norm(x_reshaped)
        gru_out, _ = self.gru(normalized_x)
        attn_scores = self.attention_layer(gru_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        context_vector = torch.sum(gru_out * attn_weights, dim=1)
        context_vector_dropped = self.shared_dropout(context_vector)

        # Regression Head
        reg_hidden = self.reg_relu(self.reg_fc1(context_vector_dropped))
        regression_scores = self.reg_output_activation(self.reg_fc2(reg_hidden))

        # Classification Head
        cls_hidden = self.cls_relu(self.cls_fc1(context_vector_dropped))
        classification_logits = self.cls_fc2(cls_hidden)

        total_loss = None
        loss_reg = None
        loss_cls = None

        if labels is not None:
            regression_targets = labels.get('regression_targets')
            classification_targets = labels.get('classification_targets')

            current_total_loss = torch.tensor(0.0, device=x.device)

            if regression_targets is not None and self.regression_loss_fn is not None:
                if regression_targets.ndim == 1:
                    regression_targets = regression_targets.unsqueeze(1)
                loss_reg = self.regression_loss_fn(regression_scores, regression_targets)
                current_total_loss += self.regression_loss_weight * loss_reg

            if classification_targets is not None and self.classification_loss_fn is not None:
                loss_cls = self.classification_loss_fn(classification_logits, classification_targets.long())
                current_total_loss += self.classification_loss_weight * loss_cls

            total_loss = current_total_loss

            # --- THIS IS THE KEY CHANGE ---
            # Prepare the core output dictionary that ALWAYS includes attention_weights
        output = {
            "regression_scores": regression_scores,
            "classification_logits": classification_logits,
            # Ensure attention_weights are always included for ONNX export
            # Squeeze the last dimension if it's 1, for potentially easier handling later
            "attention_weights": attn_weights.squeeze(-1)  # Or just attn_weights if you prefer to keep the last dim
        }

        if total_loss is not None:
            output["loss"] = total_loss
        if loss_reg is not None:
            output["loss_regression"] = loss_reg
        if loss_cls is not None:
            output["loss_classification"] = loss_cls

        return output
