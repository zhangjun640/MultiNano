import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Consider whether d_model is odd when computing div_term
        half_d_model = d_model // 2
        div_term = torch.exp(torch.arange(0, half_d_model, dtype=torch.float) * (-math.log(10000.0) / d_model))

        # Handle even d_model
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        else:
            # If d_model is odd, handle the last dimension separately
            pe[:, 1::2] = torch.cos(position * div_term)  # First part
            pe[:, -1] = torch.sin(position * div_term[-1])  # Last dimension

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SignalLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_ff, dropout=0.0):
        super().__init__()
        self.sa_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.sa_dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ff_block = nn.Sequential(
            nn.Linear(embed_dim, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, signal, signal_mask=None):
        # Self-attention block
        norm_signal = self.sa_norm(signal)
        attn_output = self.self_attn(
            norm_signal,
            norm_signal,
            norm_signal,
            key_padding_mask=signal_mask,
            need_weights=False
        )[0]
        signal = signal + self.sa_dropout(attn_output)

        # Feed forward block
        norm_signal = self.ff_norm(signal)
        ff_output = self.ff_block(norm_signal)
        signal = signal + ff_output

        return signal


class SignalEncoder(nn.Module):
    def __init__(self,
                 embed_dim=6,       # New parameter, originally fixed to 6
                 num_heads=3,       # New parameter, originally fixed to 3
                 n_layers=3,        # New parameter, originally fixed to 3
                 dim_ff=32,         # New parameter, originally fixed to 32
                 dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.layers = nn.ModuleList([
            SignalLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_ff=dim_ff,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(embed_dim, 256)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, signal, signal_mask=None):
        # signal shape: [batch_size, seq_len, embed_dim]
        signal = self.pos_encoder(signal)

        for layer in self.layers:
            signal = layer(signal, signal_mask)

        # Pool and project
        signal = self.pool(signal.transpose(1, 2))  # [batch_size, embed_dim, 1]
        signal = torch.flatten(signal, 1)  # [batch_size, embed_dim]
        signal = self.output_proj(signal)  # [batch_size, output_dim]

        return signal


class FeatureFusionModule(nn.Module):
    def __init__(self):
        super().__init__()

        # Cross-attention mechanism
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            batch_first=True  # Key parameter
        )

        # Dimension adaptation layer (with depthwise separable convolution)
        self.adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Transformer fusion with residual connections
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True  # Must be explicitly set
            ),
            num_layers=3
        )

        # Initialization
        self._init_weights()

    def _init_weights(self):
        # Xavier initialization for adapter
        nn.init.xavier_normal_(self.adapter[0].weight)
        nn.init.zeros_(self.adapter[0].bias)

        # Transformer layer initialization
        for p in self.fusion_transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, resnet_feat1, resnet_feat2):
        # Feature interaction (cross-attention)
        attn_output, _ = self.cross_attn(
            query=resnet_feat1,
            key=resnet_feat2,
            value=resnet_feat2
        )

        # Residual connection
        interacted_feat = resnet_feat1 + attn_output

        # Dimensional adaptation
        fused_feat = self.adapter(
            torch.cat([interacted_feat, resnet_feat2], dim=-1)
        )

        # Deep fusion with Transformer
        return self.fusion_transformer(fused_feat)


class Basecalling_Encoder(nn.Module):
    def __init__(self,
                 embed_dim=8,  # Input feature dimension (4+1+1+1+1=8)
                 num_heads=4,
                 n_layers=3,
                 dim_ff=64,
                 dropout=0.1):
        super().__init__()

        # Positional encoding (for sequence order)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        # Transformer encoding layers
        self.layers = nn.ModuleList([
            SignalLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_ff=dim_ff,
                dropout=dropout
            ) for _ in range(n_layers)
        ])

        # Dimensional projection module
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256)
        )

        # Temporal feature pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, kmer, qual, mis, ins, dele):
        """Adapt to multi-feature input
        Args:
            kmer: [B, S, 4]  # Embedded k-mer features
            qual/mis/ins/dele: [B, S]  # Corresponding quality scores
        """
        # Feature standardization
        features = torch.cat([
            kmer,
            qual.unsqueeze(-1),
            mis.unsqueeze(-1),
            ins.unsqueeze(-1),
            dele.unsqueeze(-1)
        ], dim=-1)  # [B, S, 8]

        # Positional encoding
        x = self.pos_encoder(features)  # [B, S, 8]

        # Transformer processing
        for layer in self.layers:
            x = layer(x)

        # Temporal feature aggregation
        x = x.mean(dim=1)  # [B, 8]

        # Dimensional expansion
        return self.projection(x)  # [B, 256]
