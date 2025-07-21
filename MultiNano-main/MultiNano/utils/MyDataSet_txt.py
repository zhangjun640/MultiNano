from torch.utils.data import Dataset
import linecache
import os
import numpy as np

from MultiNano.utils.constants import base2code, min_cov, max_seq_len, max_signal_len


def clear_linecache():
    linecache.clearcache()


def _parse_one_read_level_feature(feature, kmerb, kmere, signalb, signale, kmer):
    words = feature.strip().split("||")

    base_signal_lens = np.array([int(x) for x in words[3].split(",")])[kmerb:kmere]
    # k_signals = np.array([[float(y) for y in x.split(",")] for x in words[4].split(";")])[kmerb:kmere, signalb:signale]

    k_signals = np.array([[float(y) for y in x.split(",")] for x in words[4].split(";")])[kmerb:kmere]

    k_signals_new = np.zeros((len(k_signals), signale - signalb))
    for i in range(len(k_signals)):
        k_signals_tem = k_signals[i]
        if base_signal_lens[i] < signale - signalb:
            k_signals_tem = k_signals_tem[signalb:signale]
        else:
            pad0s = max_signal_len - base_signal_lens[i]
            padl, padr = pad0s // 2, pad0s - pad0s // 2
            k_signals_tem = k_signals_tem[padl:-padr] if pad0s > 0 else k_signals_tem
            k_signals_tem = k_signals_tem[np.sort(np.random.choice(len(k_signals_tem), signale - signalb, replace=False))]
        k_signals_new[i] = k_signals_tem
    k_signals = k_signals_new

    Mean= np.array([float(x) for x in words[0].split(",")])[kmerb:kmere]
    Median=np.array([float(x) for x in words[1].split(",")])[kmerb:kmere]
    SD=np.array([float(x) for x in words[2].split(",")])[kmerb:kmere]
    qual = np.array([int(x) for x in words[5].split(",")])[kmerb:kmere]
    mis = np.array([int(x) for x in words[6].split(",")])[kmerb:kmere]
    ins = np.array([int(x) for x in words[7].split(",")])[kmerb:kmere]
    dele = np.array([int(x) for x in words[8].split(",")])[kmerb:kmere]
    # assert len(qual) == len(mis) == len(ins) == len(dele) == len(kmer)
    return np.concatenate((kmer.reshape(-1, len(kmer)), Mean.reshape(-1, len(kmer)),Median.reshape(-1, len(kmer)),
                           SD.reshape(-1, len(kmer)),qual.reshape(-1, len(kmer)),
                           mis.reshape(-1, len(kmer)), ins.reshape(-1, len(kmer)),
                           dele.reshape(-1, len(kmer)), k_signals.transpose()),
                           axis=0)

def generate_features_line1(line, kmerb=3, kmere=8, signalb=0, signale=65, sampleing=True):
    words = line.strip().split("\t")
    sampleinfo = "\t".join(words[0:3])
    coverage = int(words[3])
    full_kmer = words[4]  # 保留原始全长 kmer
    kmer = np.array([base2code[x] for x in full_kmer])[kmerb:kmere]
    features = np.array(words[5:-1])
    assert len(features) == coverage
    if sampleing:
        if coverage < min_cov:
            indices = np.random.choice(coverage, min_cov, replace=True)
        else:
            indices = np.random.choice(coverage, min_cov, replace=False)
        features = features[indices]
    features = np.array([
        _parse_one_read_level_feature(x, kmerb, kmere, signalb, signale, kmer) for x in features
    ])
    label = int(words[-1])
    center5mer = full_kmer[len(full_kmer) // 2 - 2:len(full_kmer) // 2 + 3]
    return sampleinfo, features, center5mer


def generate_features_line(line, kmerb=3, kmere=8, signalb=0, signale=65, sampleing=True):
    words = line.strip().split("\t")
    sampleinfo = "\t".join(words[0:3])
    coverage = int(words[3])
    kmer = np.array([base2code[x] for x in words[4]])[kmerb:kmere]
    features = np.array(words[5:-1])
    assert len(features) == coverage
    if sampleing:
        if coverage < min_cov:
            # 允许重复采样以保证维度统一
            indices = np.random.choice(coverage, min_cov, replace=True)
        else:
            indices = np.random.choice(coverage, min_cov, replace=False)
        features = features[indices]
    features = np.array([_parse_one_read_level_feature(x, kmerb, kmere, signalb, signale, kmer) for x in features])
    label = int(words[-1])
    return sampleinfo, features, label


class MyDataSetTxt(Dataset):
    def __init__(self, filename, seq_len=5, signal_len=65, transform=None):
        # print(">>>using linecache to access '{}'<<<\n"
        #       ">>>after done using the file, "
        #       "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self.seq_len=seq_len
        center_pos = max_seq_len // 2
        half_len = self.seq_len // 2
        if self.seq_len % 2 == 1:
            self.kmerb = center_pos - half_len
            self.kmere = center_pos + half_len + 1
        else:
            self.kmerb = center_pos - half_len
            self.kmere = center_pos + half_len

        center_pos_signal = max_signal_len // 2
        self.signalb, self.signale = center_pos_signal - signal_len//2, center_pos_signal + signal_len//2 + 1
        self._transform = transform
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = generate_features_line(line, self.kmerb, self.kmere, self.signalb, self.signale)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data

    def close(self):
        pass


# FeaData2 ======================================================
# ChunkDataset hasn't being accepted
# https://github.com/pytorch/pytorch/pull/26547

# https://github.com/pytorch/text/issues/130
# https://github.com/pytorch/text/blob/0b4718d7827b7f278cd3169af7f2587c1f663a27/torchtext/datasets/unsupervised_learning.py
def generate_offsets(filename):
    offsets = []
    with open(filename, "r") as rf:
        offsets.append(rf.tell())
        while rf.readline():
            offsets.append(rf.tell())
    return offsets


class MyDataSetTxt2(Dataset):
    def __init__(self, filename, offsets, linenum, seq_len=5, signal_len=65, transform=None):
        self._filename = os.path.abspath(filename)
        self._total_data = linenum

        self.seq_len=seq_len
        center_pos = max_seq_len // 2
        self.kmerb, self.kmere = center_pos - self.seq_len//2, center_pos + self.seq_len//2 + 1
        center_pos_signal = max_signal_len // 2
        self.signalb, self.signale = center_pos_signal - signal_len//2, center_pos_signal + signal_len//2 + 1

        self._transform = transform

        self._offsets = offsets
        # self._data_stream = open(self._filename, 'r')
        self._current_offset = 0

    def __getitem__(self, idx):
        offset = self._offsets[idx]
        # self._data_stream.seek(offset)
        # line = self._data_stream.readline()
        with open(self._filename, "r") as rf:
            rf.seek(offset)
            line = rf.readline()
        output = generate_features_line(line, self.kmerb, self.kmere, self.signalb, self.signale)
        if self._transform is not None:
            output = self._transform(output)
        return output

    def __len__(self):
        return self._total_data

    def close(self):
        # self._data_stream.close()
        pass


def read_level_features_line(line, kmerb, kmere, signalb=0, signale=65):
    words = line.strip().split("\t")
    sampleinfo = "\t".join(words[0:6])
    try:
        kmers = np.array([base2code[x] for x in words[6]])[kmerb:kmere]
    except KeyError as e:
        print(f"[WARNING] Unexpected character '{e.args[0]}' in k-mer: {words[6]} (line skipped)")
        return None
    base_means = np.array([float(x) for x in words[7].split(",")])[kmerb:kmere]
    base_median = np.array([float(x) for x in words[8].split(",")])[kmerb:kmere]
    base_stds = np.array([float(x) for x in words[9].split(",")])[kmerb:kmere]
    base_signal_lens = np.array([int(x) for x in words[10].split(",")])[kmerb:kmere]
    # k_signals = np.array([[float(y) for y in x.split(",")] for x in words[11].split(";")])[kmerb:kmere, signalb:signale]
    k_signals = np.array([[float(y) for y in x.split(",")] for x in words[11].split(";")])[kmerb:kmere]
    k_signals_new = np.zeros((len(k_signals), signale - signalb))
    for i in range(len(k_signals)):
        k_signals_tem = k_signals[i]
        if base_signal_lens[i] < signale - signalb:
            k_signals_tem = k_signals_tem[signalb:signale]
        else:
            pad0s = max_signal_len - base_signal_lens[i]
            padl, padr = pad0s // 2, pad0s - pad0s // 2
            k_signals_tem = k_signals_tem[padl:-padr] if pad0s > 0 else k_signals_tem
            k_signals_tem = k_signals_tem[np.sort(np.random.choice(len(k_signals_tem), signale - signalb, replace=False))]
        k_signals_new[i] = k_signals_tem
    k_signals = k_signals_new

    
    qual = np.array([int(x) for x in words[12].split(",")])[kmerb:kmere]
    mis = np.array([int(x) for x in words[13].split(",")])[kmerb:kmere]
    ins = np.array([int(x) for x in words[14].split(",")])[kmerb:kmere]
    dele = np.array([int(x) for x in words[15].split(",")])[kmerb:kmere]
    try:
        labels = int(words[16])
    except IndexError:
        labels = 0


    features = np.concatenate((kmers.reshape(-1, len(kmers)), base_means.reshape(-1, len(kmers)), base_median.reshape(-1, len(kmers)),
                               base_stds.reshape(-1, len(kmers)),qual.reshape(-1, len(kmers)),
                               mis.reshape(-1, len(kmers)), ins.reshape(-1, len(kmers)), 
                               dele.reshape(-1, len(kmers)), k_signals.transpose()), 
                               axis=0)
    return sampleinfo, features, labels


class MyDataSetTxt_read_level(Dataset):
    def __init__(self, filename, seq_len=5, signal_len=65, transform=None):
        # print(">>>using linecache to access '{}'<<<\n"
        #       ">>>after done using the file, "
        #       "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self.seq_len=seq_len
        center_pos = max_seq_len // 2
        self.kmerb, self.kmere = center_pos - self.seq_len//2, center_pos + self.seq_len//2 + 1
        center_pos_signal = max_signal_len // 2
        self.signalb, self.signale = center_pos_signal - signal_len//2, center_pos_signal + signal_len//2 + 1
        self._transform = transform
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = read_level_features_line(line, self.kmerb, self.kmere, self.signalb, self.signale)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data

    def close(self):
        pass