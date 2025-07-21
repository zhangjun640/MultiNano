import torch
import torch.nn as nn


class BiLSTM_Basecaller(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, output_size=256, dropout=0.2):
        super().__init__()
        # 输入维度说明：
        # input_size = embedding_size(4) + 4个质量特征 = 8
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)  # 双向拼接后的维度
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

        # 参数初始化
        for name, param in self.bilstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    def forward(self, x):
        # 输入x形状: (batch_size, seq_len, input_size)
        # 例如: (N, 5, 8)

        # BiLSTM处理
        output, (h_n, c_n) = self.bilstm(x)

        # 使用最后一个时间步的输出
        last_output = output[:, -1, :]

        # 层归一化和Dropout
        x = self.layer_norm(last_output)
        x = self.dropout(x)

        # 全连接层映射到目标维度
        return self.fc(x)