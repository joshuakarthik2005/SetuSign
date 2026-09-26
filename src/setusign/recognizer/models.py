"""
Baseline Sign Language Recognition Models
==========================================
Provides Transformer and LSTM models for classifying keypoint sequences.
Input: [B, T, F] where B=batch, T=time, F=features (150 keypoints).
Output: [B, num_classes] logits.

The Transformer model is the primary baseline, matching the AI4Bharat approach.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: [B, T, D]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SignTransformer(nn.Module):
    """
    Transformer-based sign language recognizer.
    
    Architecture:
        Input projection (F -> d_model) -> Positional Encoding ->
        N Transformer Encoder Layers -> Mean pooling -> Classifier head
    
    This is designed to be ONNX-exportable for Snapdragon NPU deployment.
    """

    def __init__(self, input_size=150, num_classes=50, d_model=256,
                 nhead=8, num_layers=4, dim_feedforward=512,
                 dropout=0.1, max_len=200):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, F] keypoint features
            mask: [B, T] attention mask (1=valid, 0=padding)
        
        Returns:
            logits: [B, num_classes]
        """
        # Project input
        x = self.input_proj(x)  # [B, T, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create key_padding_mask for transformer (True = ignore)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)  # [B, T], True = padding

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        # Masked mean pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)  # [B, d_model]

        # Classify
        logits = self.classifier(x)  # [B, num_classes]
        return logits

    def get_encoder_output(self, x, mask=None):
        """Get encoder output before classification (for personalization)."""
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return x  # [B, d_model]


class SignLSTM(nn.Module):
    """
    Bidirectional LSTM sign language recognizer (fallback baseline).
    """

    def __init__(self, input_size=150, num_classes=50, hidden_size=256,
                 num_layers=3, dropout=0.2, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * factor, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, F] keypoint features
            mask: [B, T] attention mask (1=valid, 0=padding)
        Returns:
            logits: [B, num_classes]
        """
        output, (hn, cn) = self.lstm(x)  # output: [B, T, H*2]

        # Masked mean pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            output = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            output = output.mean(dim=1)

        logits = self.classifier(output)
        return logits


def build_model(model_type="transformer", input_size=150, num_classes=50, **kwargs):
    """Factory function to create a model."""
    if model_type == "transformer":
        return SignTransformer(input_size=input_size, num_classes=num_classes, **kwargs)
    elif model_type == "lstm":
        return SignLSTM(input_size=input_size, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
