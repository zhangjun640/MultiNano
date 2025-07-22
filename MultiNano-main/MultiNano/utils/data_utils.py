#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Demo_xu_4.4 
@File    ：data_utils.py
@Author  ：XJR
@Date    ：2022/4/4 17:15 
'''
import os
import fnmatch
from itertools import product
import torch
# from utils.MyDataSet import MyDataSet_train, MyDataSet, MyDataSet_val
from torch.utils.data import Dataset, DataLoader
from MultiNano.utils.MyDataSet_txt import MyDataSetTxt

def get_fast5_files(fast5_dir, is_recursive=True):
    fast5_dir = os.path.abspath(fast5_dir) # get abs directoty
    fast5_files = []
    if is_recursive:
        for root, dirnames, filenames in os.walk(fast5_dir):
            for filename in fnmatch.filter(filenames, "*fast5"):
                fast5_path = os.path.join(root, filename)
                fast5_files.append(fast5_path)
    else:
        for filename in os.listdir(fast5_dir):
            if filename.endswith(".fast5"):
                fast5_path = '/'.join([fast5_dir, filename])
                fast5_files.append(fast5_path)
    return fast5_files

def get_motifs(motifs='DRACH'):
    if motifs not in ['DRACH','RRACH']:
        print("cannot identify moitfs! (Motifs in ['DRACH','RRACH']) ")
    if motifs=='DRACH':
        center_motifs = [['A', 'G', 'T'], ['G', 'A'], ['A'], ['C'], ['A', 'C', 'T']]
    elif motifs =='RRACH':
        center_motifs = [['A', 'G'], ['G', 'A'], ['A'], ['C'], ['A', 'C']]
    all_kmers = list(["".join(x) for x in product(*(center_motifs))])
    return all_kmers

def makedirs(main_dir, sub_dirs=None, opt='depth'):
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    filepaths = dict()
    if sub_dirs is not None:
        if opt == 'depth':
            path = main_dir
            for sub_dir in sub_dirs:
                path = os.path.join(path, sub_dir)
                filepaths[sub_dir] = path
                # if not os.path.exists(path):
                try:  # Use try-catch for the case of multiprocessing.
                    os.makedirs(path)
                except:
                    pass

        else:  # opt == 'breadth'
            for sub_dir in sub_dirs:
                path = os.path.join(main_dir, sub_dir)
                filepaths[sub_dir] = path
                # if not os.path.exists(path):
                try:  # Use try-catch for the case of multiprocessing.
                    os.makedirs(path)
                except:
                    pass
    return filepaths

def base_embedding():
    return

def dataloader_split(features_file, device, batch_size=128):
    dataset = MyDataSetTxt(features_file)
    print("dataset len", len(dataset))
    print("dataloader batch size ", batch_size)
    train_size = int(len(dataset) * 0.8)
    validate_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - validate_size - train_size

    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                  [train_size, validate_size,
                                                                                   test_size])

    # torch.save(train_dataset, "train_dataset.pt")
    print("train_dataset len", len(train_dataset))
    print("validate_dataset len", len(validate_dataset))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=30)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=True, num_workers=30)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=30)

    # dataiter = iter(train_loader)
    # data = datait
    # er.next()
    # print(train_loader.batch_size)
    return train_loader, validate_loader, test_loader


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input


def count_line_num(sl_filepath, fheader=False):
    count = 0
    with open(sl_filepath, 'r') as rf:
        if fheader:
            next(rf)
        for _ in rf:
            count += 1
    print('done count the lines of file {}'.format(sl_filepath))
    return count
