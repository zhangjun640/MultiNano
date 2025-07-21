import os
import torch
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys

sys.path.append(os.getcwd())
import time

from MultiNano.model.modelOneGASF import SiteLevelModel
from MultiNano.utils.constants import max_seq_len, use_cuda, FloatTensor, min_cov, max_signal_len
from MultiNano.utils.MyDataSet_txt import generate_features_line1

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
    # 32 14 0.8488

    # 16 17 0.8557
    """

    """
    parser.add_argument("--model", default=r"/root/autodl-tmp/111111111transfer/arab/model/jun_all2/early_stopping_model_states.pt",
                        required=False, help="model file for test")
    parser.add_argument("--input_file", default=r"/root/autodl-tmp/hek293t/train/test0.txt", required=False,
                        help="input features file")
    # /root/autodl-tmp/syn/test/mod_rep2_11mer_65_features.label.sort.aggre.c15.c20_1000.tsv
    # /root/autodl-tmp/syn/test/unmod_rep2_11mer_65_features.label.sort.aggre.c15.c20_1000.tsv
    parser.add_argument("--seed", default=44, type=int)  # 44
    parser.add_argument("--num_iterations", default=5, type=int)
    parser.add_argument("--batch_size", default=3, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)

    parser.add_argument("--seq_lens", default=5, type=int)
    parser.add_argument("--signal_lens", default=65, type=int)
    parser.add_argument("--embedding_size", default=4, type=int)
    parser.add_argument("--model_type", default='comb_basecall_raw', type=str,
                        choices=["basecall", "raw_signal", "comb_basecall_raw"],
                        required=False, help="module for train:[basecall, raw_signal, comb_basecall_raw]")

    parser.add_argument("--output_file", default=r"/root/autodl-tmp/111111111transfer/syn/test/nano/label0_000000.tsv",required=False, type=str, help="output file")

    parser.add_argument("--nproc", default=4, type=int, help="number of processes")

    return parser


def read_feature_file(feature_file, features_batch_q, batch_size=2, seq_len=5, signal_len=65):
    print("read_features process-{} starts".format(os.getpid()))
    center_pos = max_seq_len // 2
    kmerb, kmere = center_pos - seq_len // 2, center_pos + seq_len // 2 + 1
    center_pos_signal = max_signal_len // 2
    signalb, signale = center_pos_signal - signal_len // 2, center_pos_signal + signal_len // 2 + 1
    sampleids, center5mers, features = [], [], []
    site_num, site_batch = 0, 0
    with open(feature_file, "r") as f:
        for line in f:
            site_num += 1
            sampleid, feature, center5mer = generate_features_line1(line, kmerb, kmere, signalb, signale, False)
            sampleids.append(sampleid)
            center5mers.append(center5mer)
            features.append(feature)
            if len(features) == batch_size:
                features_batch_q.put((sampleids, center5mers, features))
                site_batch += 1
                while features_batch_q.qsize() > queue_size_border_batch:
                    time.sleep(0.1)
                sampleids, center5mers, features = [], [], []
        if len(features) > 0:
            features_batch_q.put((sampleids, center5mers, features))
    features_batch_q.put("kill")
    print("read_features process-{} ending, read {} site_features in {} batches({})".format(
        os.getpid(), site_num, site_batch, batch_size))

def _predict(features_batch, model, num_iters=5, device=0):
    sampleids, center5mers, features = features_batch
    covs = [x.shape[0] for x in features]
    covs_cumsum = np.cumsum(covs)
    features_all = FloatTensor(np.concatenate(features, axis=0), device)
    read_probs = model.get_read_level_probs(features_all)
    if use_cuda:
        read_probs = read_probs.detach().cpu()
    read_probs = read_probs.numpy().flatten()
    read_prob_groups = np.split(read_probs, covs_cumsum[:-1])
    assert len(read_prob_groups) == len(sampleids) and sum(len(x) for x in read_prob_groups) == len(features_all)
    pred_str = []
    for idx in range(len(read_prob_groups)):
        site_prob = np.concatenate([
            np.random.choice(read_prob_groups[idx], min_cov, replace=False)
            for _ in range(num_iters)
        ]).reshape(num_iters, min_cov)
        site_prob_mean = (1 - np.prod(1 - site_prob, axis=1)).mean()
        label = 1 if site_prob_mean >= 0.5 else 0
        pred_str.append("\t".join([sampleids[idx], center5mers[idx], str(round(site_prob_mean, 6)), str(label)]))
    return pred_str



def predict(model_path, features_batch_q, pred_str_q, args, device=0):
    print('call_mods process-{} starts'.format(os.getpid()))
    model = SiteLevelModel(args.model_type, 0, args.hidden_size,
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
        pred_str = _predict(features_batch, model, args.num_iterations, device)
        pred_str_q.put(pred_str)
        while pred_str_q.qsize() > queue_size_border_batch:
            time.sleep(0.1)
    print('call_mods process-{} ending, process {} batches({})'.format(os.getpid(), batch_num,
                                                                       args.batch_size))

def _write_predstr_to_file(write_fp, predstr_q):
    print('write_process-{} starts'.format(os.getpid()))
    with open(write_fp, 'w') as wf:
        while True:
            # during test, it's ok without the sleep()
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
    print("[main]predicting starts..")
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
    p_rf = mp.Process(target=read_feature_file, args=(args.input_file, features_batch_q, batch_size,
                                                      args.seq_lens, args.signal_lens))
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
    print("[main]predicting costs %.2f seconds.." % (time.time() - start))


def main():
    args = argparser().parse_args()
    test(args)


if __name__ == '__main__':
    main()