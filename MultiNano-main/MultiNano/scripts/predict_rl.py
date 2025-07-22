import os
import torch
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
sys.path.append(os.getcwd())
import time

from MultiNano.model.modelOneGASF import ReadLevelModel
from MultiNano.utils.constants import max_seq_len, use_cuda, FloatTensor, code2base
from MultiNano.utils.MyDataSet_txt import read_level_features_line

import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass
from torch.multiprocessing import Queue

queue_size_border_batch = 100


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("--model", required=True, help="model file for test")
    parser.add_argument("--input_file",  required=True, help="input features file")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)

    parser.add_argument("--seq_lens", default=5, type=int)
    parser.add_argument("--signal_lens", default=65, type=int)
    parser.add_argument("--embedding_size", default=4, type=int)
    parser.add_argument("--model_type", default='comb', type=str,
                        choices=["basecall", "raw_signals", "comb"],
                        required=False, help="module for train:[basecall, raw_signal, comb_basecall_raw]")
                        
    parser.add_argument("--output_file", default="/root/autodl-tmp/1111New/syn/nano_syn0.tsv",required=False, type=str, help="output file")

    parser.add_argument("--nproc", default=4, type=int, help="number of processes")

    return parser


def read_feature_file(feature_file, features_batch_q, batch_size=2, seq_len=5):
    print("read_features process-{} starts".format(os.getpid()))
    center_pos = max_seq_len // 2
    kmerb, kmere = center_pos - seq_len//2, center_pos + seq_len//2 + 1
    sampleids, features = [], []
    site_num, site_batch = 0, 0
    with open(feature_file, "r") as f:
        for line in f:
            site_num += 1
            sampleid, feature, _ = read_level_features_line(line, kmerb, kmere)
            sampleids.append(sampleid)
            features.append(feature)
            if len(features) == batch_size:
                features_batch_q.put((sampleids, np.array(features)))
                site_batch += 1
                while features_batch_q.qsize() > queue_size_border_batch:
                    time.sleep(0.1)
                sampleids, features = [], []
        if len(features) > 0:
            features_batch_q.put((sampleids, np.array(features)))
    features_batch_q.put("kill")
    print("read_features process-{} ending, read {} sample_features in {} batches({})".format(os.getpid(),
                                                                                              site_num,
                                                                                              site_batch,
                                                                                              batch_size))


def _predict(features_batch, model, device=0):
    sampleids, features = features_batch
    kmers = features[:, 0, :]
    features_all = FloatTensor(features, device)
    read_probs = model(features_all)
    if use_cuda:
        read_probs = read_probs.detach().cpu()
    read_probs = read_probs.numpy().flatten()
    pred_str = []
    for idx in range(len(read_probs)):
        prob = round(read_probs[idx], 6)
        meth = 1 if prob >= 0.5 else 0
        pred_str.append("\t".join([
            sampleids[idx],
            "".join([code2base[x] for x in kmers[idx]]),
            str(prob),
            str(meth)
        ]))
    return pred_str


def predict(model_path, features_batch_q, pred_str_q, args, device=0):
    print('call_mods process-{} starts'.format(os.getpid()))
    model = ReadLevelModel(args.model_type, 0, args.hidden_size, 
                           args.seq_lens, args.signal_lens, args.embedding_size, 
                           device=device)
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(checkpoint['net'])
    except RuntimeError:
        model.module.load_state_dict(checkpoint['net'])
    if use_cuda:
        model = model.cuda(device)
    model.eval()
    batch_num = 0
    while True:
        if features_batch_q.empty():
            time.sleep(0.1)
            continue
        features_batch = features_batch_q.get()
        if features_batch == "kill":
            features_batch_q.put("kill")
            break
        batch_num += 1
        pred_str = _predict(features_batch, model, device)
        pred_str_q.put(pred_str)
        while pred_str_q.qsize() > queue_size_border_batch:
            time.sleep(0.1)
    print('call_mods process-{} ending, process {} batches({})'.format(os.getpid(), batch_num, 
                                                                       args.batch_size))


def _write_predstr_to_file(write_fp, predstr_q):
    print('write_process-{} starts'.format(os.getpid()))
    with open(write_fp, 'w') as wf:
        wf.write("chrom\tpos\talignstrand\tloc_in_re\treadname\tstrand\tkmer\tprob\tmethylation\n")
        while True:
            if predstr_q.empty():
                time.sleep(0.1)
                continue
            pred_str = predstr_q.get()
            if pred_str == "kill":
                print('write_process-{} finished'.format(os.getpid()))
                break
            for one_pred_str in pred_str:
                wf.write(one_pred_str + "\n")
            wf.flush()


def _get_gpus():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpulist = list(range(num_gpus))
    else:
        gpulist = [0]
    return gpulist * 1000


def test(args):
    print("[main]read-level predicting starts..")
    start = time.time()
    if not os.path.exists(args.model):
        raise FileNotFoundError("model file not exists")

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    nproc = args.nproc
    if nproc < 3:
        print("nproc should be larger than 2")
        nproc = 3
    nproc_dp = nproc - 2
    if not use_cuda:
        nproc_dp = 2

    batch_size = args.batch_size
    features_batch_q = Queue()
    p_rf = mp.Process(target=read_feature_file, args=(args.input_file, features_batch_q, batch_size, args.seq_lens))
    p_rf.daemon = True
    p_rf.start()

    pred_str_q = Queue()
    predstr_procs = []
    gpulist = _get_gpus()
    gpuindex = 0
    for _ in range(nproc_dp):
        p = mp.Process(target=predict, args=(args.model, features_batch_q, pred_str_q, args, gpulist[gpuindex]))
        p.daemon = True
        p.start()
        predstr_procs.append(p)
        gpuindex += 1
    
    p_w = mp.Process(target=_write_predstr_to_file, args=(args.output_file, pred_str_q))
    p_w.daemon = True
    p_w.start()

    for p in predstr_procs:
        p.join()
    
    pred_str_q.put("kill")
    p_rf.join()
    p_w.join()
    print("[main]read-level predicting costs %.2f seconds.." % (time.time() - start))


def main():
    args = argparser().parse_args()
    test(args)


if __name__ == '__main__':
    main()
