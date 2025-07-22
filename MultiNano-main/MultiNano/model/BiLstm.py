import torch
import torch.nn as nn


class BiLSTM_Basecaller(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, output_size=256, dropout=0.2):
        super().__init__()
        # Input dimension explanation:
        # input_size = embedding_size (4) + 4 quality features = 8
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)  # Dimension after bidirectional concatenation
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

        # Parameter initialization
        for name, param in self.bilstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    def forward(self, x):
        # Input x shape: (batch_size, seq_len, input_size)
        # Example: (N, 5, 8)

        # BiLSTM processing
        output, (h_n, c_n) = self.bilstm(x)

        # Use the output from the last time step
        last_output = output[:, -1, :]

        # Layer normalization and Dropout
        x = self.layer_norm(last_output)
        x = self.dropout(x)

        # Fully connected layer mapping to target dimension
        return self.fc(x)
