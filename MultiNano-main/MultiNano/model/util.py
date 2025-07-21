import torch
from torch import nn

# Creating Fully Connected Network
class Full_NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super(Full_NN, self).__init__()
        self.fc1 = nn.Linear(input_size,  hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.dropout(self.fc1(x))
        x = self.relu(x)
        x = self.fc2(x)
        return x


class FlattenLayer(torch.nn.Module):
    def __init__(self):
            super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def one_hot_embedding(kmer, signal_lens):
    # print(kmer.shape) # torch.Size([N, 5])
    # print(signal_lens)
    # print("kmer", kmer[0])

    expand_kmer = torch.repeat_interleave(kmer, signal_lens, dim=1)
    # print("DD", expand_kmer.shape) # torch.Size([N, 150])

    # expand_kmer  = expand_kmer.view(kmer.shape[0], -1, signal_lens)
    # print("DD", expand_kmer.shape) # torch.Size([N, 5, 30])

    embed = nn.functional.one_hot(expand_kmer, 4)
    # print(embed.shape) # torch.Size([N, 5, 30 ,4])

    return  embed

