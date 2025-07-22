import torch

use_cuda = torch.cuda.is_available()

def FloatTensor(tensor, device=0):
    if use_cuda:
        return torch.tensor(tensor, dtype=torch.float, device='cuda:{}'.format(device))
    return torch.tensor(tensor, dtype=torch.float)

base2code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
code2base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

min_cov = 20

max_seq_len = 11

max_signal_len = 65

