import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算div_term时考虑d_model是否为奇数
        half_d_model = d_model // 2
        div_term = torch.exp(torch.arange(0, half_d_model, dtype=torch.float) * (-math.log(10000.0) / d_model))

        # 处理偶数d_model
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        else:
            # 如果d_model为奇数，处理最后一个维度
            pe[:, 1::2] = torch.cos(position * div_term)  # 前半部分
            pe[:, -1] = torch.sin(position * div_term[-1])  # 后半部分单独处理

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
                 embed_dim=6,       # 新增参数，原固定值6
                 num_heads=3,       # 新增参数，原固定值3
                 n_layers=3,        # 新增参数，原固定值3
                 dim_ff=32,         # 新增参数，原固定值32
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

        # 交叉注意力机制
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            batch_first=True  # 关键参数
        )

        # 维度适配层 (带深度可分离卷积)
        self.adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 带残差的Transformer融合
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True  # 必须显式设置
            ),
            num_layers=3
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        # Xavier初始化适配层
        nn.init.xavier_normal_(self.adapter[0].weight)
        nn.init.zeros_(self.adapter[0].bias)

        # Transformer层初始化
        for p in self.fusion_transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, resnet_feat1, resnet_feat2):
        # 特征交互 (交叉注意力)
        attn_output, _ = self.cross_attn(
            query=resnet_feat1,
            key=resnet_feat2,
            value=resnet_feat2
        )

        # 残差连接
        interacted_feat = resnet_feat1 + attn_output

        # 维度适配
        fused_feat = self.adapter(
            torch.cat([interacted_feat, resnet_feat2], dim=-1)
        )

        # Transformer深度融合
        return self.fusion_transformer(fused_feat)

class Basecalling_Encoder(nn.Module):
    def __init__(self,
                 embed_dim=8,  # 输入特征维度 (4+1+1+1+1=8)
                 num_heads=4,
                 n_layers=3,
                 dim_ff=64,
                 dropout=0.1):
        super().__init__()

        # 位置编码（处理序列顺序）
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        # Transformer编码层
        self.layers = nn.ModuleList([
            SignalLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_ff=dim_ff,
                dropout=dropout
            ) for _ in range(n_layers)
        ])

        # 维度适配模块
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256)
        )

        # 时序特征池化
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, kmer, qual, mis, ins, dele):
        """适配多特征输入
        Args:
            kmer: [B, S, 4]  # 嵌入后的kmer特征
            qual/mis/ins/dele: [B, S]  # 对应质量分数
        """
        # 特征标准化
        features = torch.cat([
            kmer,
            qual.unsqueeze(-1),
            mis.unsqueeze(-1),
            ins.unsqueeze(-1),
            dele.unsqueeze(-1)
        ], dim=-1)  # [B, S, 8]

        # 位置编码
        x = self.pos_encoder(features)  # [B, S, 8]

        # Transformer处理
        for layer in self.layers:
            x = layer(x)

        # 时序特征聚合
        x = x.mean(dim=1)  # [B, 8]

        # 维度提升
        return self.projection(x)  # [B,256]
'''

def main():
    """综合测试函数，包含以下验证内容：
    1. 模型初始化验证
    2. 前向传播维度验证
    3. 参数冻结检查
    4. 位置编码模式验证
       print("=== 开始系统测试 ===")

    # 测试1: 基础功能验证
    try:
        test_signal_encoder()
        print("[通过] 基础维度验证")
    except Exception as e:
        print(f"[失败] 基础测试: {str(e)}")

    # 测试2: 参数冻结验证
    encoder = SignalEncoder()
    param_counts = sum(p.numel() for p in encoder.parameters())
    assert param_counts == 2250, f"参数总数应为2250，实际{param_counts}"
    print("[通过] 参数冻结验证")

    # 测试3: 修正后的位置编码模式验证
    pe = encoder.pos_encoder.pe.squeeze(0)  # [max_len, d_model]
    max_len, d_model = pe.shape

    # 重新计算期望值
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    half_d_model = d_model // 2
    div_term = torch.exp(torch.arange(0, half_d_model, dtype=torch.float) * (-math.log(10000.0) / d_model))

    expected_pe = torch.zeros(max_len, d_model)
    expected_pe[:, 0::2] = torch.sin(position * div_term)
    expected_pe[:, 1::2] = torch.cos(position * div_term)

    # 处理奇数维度的情况（当前d_model=6为偶数，可省略）
    if d_model % 2 != 0:
        expected_pe[:, -1] = torch.sin(position * div_term[-1])

    assert torch.allclose(pe, expected_pe, atol=1e-6), "位置编码模式错误"
    print("[通过] 位置编码模式验证")




    """

    # 新增测试用例
    bc_encoder = Basecalling_Encoder()
    dummy_kmer = torch.randint(0, 4, (3, 100))
    dummy_qual = torch.randn(3, 100, 1)
    output = bc_encoder(dummy_kmer, dummy_qual, dummy_qual, dummy_qual, dummy_qual)
    print("测试输出维度:", output.shape)

if __name__ == "__main__":
    main()

'''
