import torch
from torch import nn
from .util import Full_NN, FlattenLayer, one_hot_embedding
from MultiNano.utils.constants import min_cov
from .transformer import SignalEncoder,Basecalling_Encoder,FeatureFusionModule
from .resnet2 import OptimizedResNet2D
from .GASF import GASF,GADF,MTF,RP
import torch.nn.functional as F
from .BiLstm import BiLSTM_Basecaller
from .resnet1 import ResNet1D

class ReadLevelModel(nn.Module):
    def __init__(self, model_type, dropout_rate=0.5, hidden_size=128,
                 seq_len=5, signal_lens=65, embedding_size=4, device=0):
        super().__init__()
        self.seq_len = seq_len
        self.relu = nn.ReLU()
        self.gadf = GADF()
        self.gasf = GASF()
        self.mtf=MTF()
        self.rp=RP()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.model_type = model_type
        self.fusion_transformer = FeatureFusionModule()
        self.resnet_fusion_adapter = nn.Linear(1024, 256)

        self.resnet2d = OptimizedResNet2D(in_channels=5, out_channels=256)
        self.raw, self.basecall = False, False
        if self.model_type in ["raw_signals", "comb"]:
            self.raw = True
        if self.model_type in ["basecall", "comb"]:
            self.basecall = True

        if self.basecall:
            self.embed = nn.Embedding(4, embedding_size)
            self.basecall_net = BiLSTM_Basecaller(
                input_size=embedding_size + 4,
                output_size=256,
                dropout=dropout_rate
            )
        if self.raw:
            self.signal_encoder = SignalEncoder(
                dropout=dropout_rate
            )
            self.resnet_r = ResNet1D(in_channels=4 + 1, out_channels=256)

        if self.raw and self.basecall:
            self.full = Full_NN(input_size=256 + 256 + 256, hidden_size=hidden_size, num_classes=1, dropout_rate=dropout_rate)
        elif self.raw:  # only raw
            self.full = Full_NN(input_size=256 + 256, hidden_size=hidden_size, num_classes=1, dropout_rate=dropout_rate)
        else:  # only basecall
            self.full = Full_NN(input_size=256, hidden_size=hidden_size, num_classes=1, dropout_rate=dropout_rate)

        # Debug: Print initialization parameters
        print(f"Initialized ReadLevelModel with raw={self.raw} and basecall={self.basecall}")

    def forward(self, features):
        # Extract different features
        kmer = features[:, 0, :]
        Mean = features[:, 1, :]
        Median = features[:, 2, :]
        SD = features[:, 3, :]
        qual = features[:, 4, :]
        mis = features[:, 5, :]
        ins = features[:, 6, :]
        dele = features[:, 7, :]

        signals = torch.transpose(features[:, 8:, :], 1, 2)


        if self.basecall:
            y_kmer_embed = self.embed(kmer.long())
            #Mean = torch.reshape(Mean, (-1, self.seq_len, 1)).float()
            #Median = torch.reshape(Median, (-1, self.seq_len, 1)).float()
            #SD = torch.reshape(SD, (-1, self.seq_len, 1)).float()
            qual = torch.reshape(qual, (-1, self.seq_len, 1)).float()
            mis = torch.reshape(mis, (-1, self.seq_len, 1)).float()
            ins = torch.reshape(ins, (-1, self.seq_len, 1)).float()
            dele = torch.reshape(dele, (-1, self.seq_len, 1)).float()
            y = torch.cat((y_kmer_embed,qual, mis, ins, dele), 2)  # (N, 5, 8)

            # Debug: Print shape after concatenation
            # y shape after concatenation:  torch.Size([1540, 5, 8])
            # y shape after concatenation:  torch.Size([1000, 5, 8])
            # print("y shape after concatenation: ", y.shape)#y shape after concatenation:  torch.Size([2560, 5, 8])

            y = self.basecall_net(y)

        if self.raw:

            signals = signals.float()
            signal_trans1 = self.gasf(signals)
            signals_len = signals.shape[2]

            #signal_trans2 = self.gadf(signals)
            #signal_trans4 = self.rp(signals)

            #signal_trans1 = F.instance_norm(signal_trans1)
            #signal_trans2 = F.instance_norm(signal_trans2)
            #signal_trans4 = F.instance_norm(signal_trans4)

            combined_resnet = torch.cat([signal_trans1], dim=1)  # [N, 512]

            combined_resnet = self.resnet2d(combined_resnet)

            kmer_embed = one_hot_embedding(kmer.long(), signals_len)  # torch.Size([N, seq_len*signal_len, 4])
            signals_ex = signals.reshape(signals.shape[0], -1, 1)

            x = torch.cat((kmer_embed, signals_ex), -1)  # (N, L, C)


            x = torch.transpose(x, 1, 2)
            x = self.resnet_r(x)

            x = torch.cat([x,combined_resnet], dim=1)

        ##################### Full connect layer
        if self.raw and self.basecall:
            z = torch.cat((x, y), 1)

            #print("z shape before Full_NN (raw + basecall): ,
            #     z.shape)  # z shape before Full_NN (raw + basecall):  torch.Size([2560, 512])

            z = self.full(z)
        elif self.raw:  # only raw
            z = self.full(x)

            # Debug: Print shape before Full_NN (raw only)
            #print("z shape before Full_NN (raw only): ", z.shape)
        else:  # basecall only
            z = self.full(y)

            # Debug: Print shape before Full_NN (basecall only)
            #print("z shape before Full_NN (basecall only): ", z.shape)

            ##################### sigmoid
        out_ = self.sigmoid(z)

        return out_
"""
        if self.raw:
            signals = signals.float()
            signals_len = signals.shape[2]
            kmer_embed = one_hot_embedding(kmer.long(), signals_len)  # torch.Size([N, seq_len*signal_len, 4])
            signals_ex = signals.reshape(signals.shape[0], -1, 1)

            # Debug: Print shape of the embeddings
            #kmer_embed shape:  torch.Size([1540, 325, 4])
            #kmer_embed shape:  torch.Size([1000, 325, 4])
            print("kmer_embed shape: ", kmer_embed.shape)#kmer_embed shape:  torch.Size([2560, 325, 4])
            #signals_ex shape:  torch.Size([1540, 325, 1])
            #signals_ex shape:  torch.Size([1000, 325, 1])
            print("signals_ex shape: ", signals_ex.shape)#signals_ex shape:  torch.Size([2560, 325, 1])

            x = torch.cat((kmer_embed, signals_ex), -1)  # (N, L, C)

            # Debug: Print shape before ResNet
            #x shape before ResNet:  torch.Size([1540, 325, 5])
            #x shape before ResNet:  torch.Size([1000, 325, 5])
            print("x shape before ResNet: ", x.shape)#x shape before ResNet:  torch.Size([2560, 325, 5])

            x = torch.transpose(x, 1, 2)
            x = self.resnet_r(x)

            # Debug: Print shape after ResNet
            #x shape after ResNet:  torch.Size([1540, 256])
            #x shape after ResNet:  torch.Size([1000, 256])
            print("x shape after ResNet: ", x.shape)#x shape after ResNet:  torch.Size([2560, 256])
"""


class SiteLevelModel(nn.Module):

    def __init__(self, model_type, dropout_rate=0.5, hidden_size=128,
                 seq_len=5, signal_lens=65, embedding_size=4, device=0):
        super(SiteLevelModel, self).__init__()
        self.read_level_model = ReadLevelModel(model_type, dropout_rate, hidden_size, seq_len, signal_lens,
                                               embedding_size, device=device)

    def get_read_level_probs(self, features):  # flattened features (N, 70, 5)
        return self.read_level_model(features)

    def forward(self, features):
        # (N, 20, 5+65, 5) -> (N, 20, 1) -> (N, 1)
        features = features.view(-1, features.shape[2], features.shape[3])


        probs = self.read_level_model(features).view(-1, min_cov)



        return 1 - torch.prod(1 - probs, axis=1)
