import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .dropout import Dropout
from .multihead_attn import MultiheadAttention

class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-5, elementwise_affine=True)
        self.dropout_module = Dropout(args.dropout)
        self.activation_fn =  F.relu
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        self.activation_dropout_module = Dropout(float(activation_dropout_p))
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = nn.Linear(
            self.embed_dim,
            args.encoder_ffn_embed_dim
        )
        self.fc2 = nn.Linear(
            args.encoder_ffn_embed_dim,
            self.embed_dim
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-5, elementwise_affine=True)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask = None):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x
    

