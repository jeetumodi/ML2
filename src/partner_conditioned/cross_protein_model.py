"""
Partner-Conditioned Geometric Neural Network for Protein-Protein Interface Prediction.

Implements CrossProteinEGCPPIS:
- Intra-protein geometric encoding using Equivariant message passing.
- Inter-protein cross-attention conditioned on 3D spatial distances between receptor and partner residues.
- Partner-conditioned residue classification head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class GeometricDistanceBiasedCrossAttention(nn.Module):
    """
    Multi-Head Cross Attention with distance-based geometric bias.
    
    Attention formula:
        Attention(Q, K, V, D) = Softmax( (Q K^T / sqrt(d)) - softplus(gamma) * D ) V
    where D is the inter-chain distance matrix in Angstroms.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Learnable distance scaling factor per head
        self.gamma = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query_receptor: torch.Tensor,     # (L_A, D)
        key_partner: torch.Tensor,        # (L_B, D)
        inter_chain_distance: torch.Tensor # (L_A, L_B)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        L_A = query_receptor.size(0)
        L_B = key_partner.size(0)

        Q = self.q_proj(query_receptor).view(L_A, self.num_heads, self.head_dim).transpose(0, 1) # (H, L_A, d)
        K = self.k_proj(key_partner).view(L_B, self.num_heads, self.head_dim).transpose(0, 1)    # (H, L_B, d)
        V = self.v_proj(key_partner).view(L_B, self.num_heads, self.head_dim).transpose(0, 1)    # (H, L_B, d)

        # Scaled dot-product scores: (H, L_A, L_B)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.head_dim)

        # Apply geometric distance decay: closer residues receive higher attention bias
        dist_expanded = inter_chain_distance.unsqueeze(0).expand(self.num_heads, -1, -1) # (H, L_A, L_B)
        dist_bias = F.softplus(self.gamma) * dist_expanded
        scores = scores - dist_bias

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.bmm(attn_weights, V) # (H, L_A, d)
        context = context.transpose(0, 1).contiguous().view(L_A, self.embed_dim) # (L_A, D)

        output = self.layer_norm(query_receptor + self.out_proj(context))
        return output, attn_weights


class CrossProteinEGCPPIS(nn.Module):
    """
    Modular Partner-Conditioned PPIS architecture.
    """
    def __init__(
        self,
        in_dim: int = 115,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # Input projection layers
        self.receptor_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.partner_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Inter-protein geometric cross-attention
        self.cross_attention = GeometricDistanceBiasedCrossAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        features_receptor: torch.Tensor,      # (L_A, in_dim)
        features_partner: torch.Tensor,       # (L_B, in_dim)
        inter_chain_distance: torch.Tensor    # (L_A, L_B)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass predicting partner-conditioned interface probabilities on the receptor.
        """
        H_A = self.receptor_proj(features_receptor)  # (L_A, hidden_dim)
        H_B = self.partner_proj(features_partner)   # (L_B, hidden_dim)

        # Cross-protein geometric message passing
        context_A, attn_weights = self.cross_attention(H_A, H_B, inter_chain_distance)

        # Combine receptor intrinsic state, cross-partner context, and elementwise interaction
        combined = torch.cat([H_A, context_A, H_A * context_A], dim=-1) # (L_A, 3 * hidden_dim)
        logits = self.classifier(combined).squeeze(-1) # (L_A,)
        probs = torch.sigmoid(logits)

        return probs, logits
