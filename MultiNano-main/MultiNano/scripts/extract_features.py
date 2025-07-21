
import time
from collections import defaultdict
from multiprocessing import Queue
import h5py
import argparse
import sys,os
import multiprocessing as mp
import pandas as pd
import numpy as np
from statsmodels import robust
import random
import fcntl

sys.path.append(os.getcwd())
from MultiNano.utils.data_utils import get_fast5_files, get_motifs
from tqdm import tqdm
sys.path.append(os.getcwd())
reads_group = '/Raw/Reads'
queen_size_border = 2000
time_wait = 3

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)


def _get_label_raw(fast5_fn, correct_group, basecall_subgroup):
    """
    get mapped raw signals with sequence from fast file
    :param fast5_fn: fast5 file path
    :param correct_group: tombo resquiggle group
    :param correct_subgroup: for template or complement
    :return: raw signals, event table
    """
    try:
        fast5_data = h5py.File(fast5_fn, 'r')
    except IOError:
        raise IOError('Error opening file. Likely a corrupted file.')
    # Get raw data
    try:
        raw_dat = list(fast5_data[reads_group].values())[0]
        # raw_attrs = raw_dat.attrs
        # raw_dat = raw_dat['Signal'].value
        raw_dat = raw_dat['Signal'][()]
    except Exception:
        raise RuntimeError('Raw data is not stored in Raw/Reads/Read_[read#] so '
                           'new segments cannot be identified.')

    # Get Events
    try:
        event = fast5_data['/Analyses/' + correct_group + '/' + basecall_subgroup + '/Events']
    except Exception:
        raise RuntimeError('events not found.')

    try:
        corr_attrs = dict(list(event.attrs.items()))
        read_start_rel_to_raw = corr_attrs['read_start_rel_to_raw']
        # print('read_start_rel_to_raw: ',read_start_rel_to_raw)
        starts = list(map(lambda x: x + read_start_rel_to_raw, event['start']))
    except KeyError:
        # starts = list(map(lambda x: x, event['start']))
        raise KeyError('no read_start_rel_to_raw in event attributes')

    starts = list(map(lambda x: x + read_start_rel_to_raw, event['start']))
    lengths = event['length'].astype(np.int_)
    base = [x.decode("UTF-8") for x in event['base']]
    assert len(starts) == len(lengths)
    assert len(lengths) == len(base)
    events = list(zip(starts, lengths, base))
    return raw_dat, events

def _get_alignment_attrs_of_each_strand(strand_path, h5obj):
    """
    :param strand_path: template or complement
    :param h5obj: h5py object
    :return: alignment information
    """
    strand_basecall_group_alignment = h5obj['/'.join([strand_path, 'Alignment'])]
    alignment_attrs = strand_basecall_group_alignment.attrs
    # attr_names = list(alignment_attrs.keys())

    if strand_path.endswith('template'):
        strand = 't'
    else:
        strand = 'c'
    try:
        alignstrand = str(alignment_attrs['mapped_strand'], 'utf-8')
        chrom = str(alignment_attrs['mapped_chrom'], 'utf-8')
    except TypeError:
        alignstrand = str(alignment_attrs['mapped_strand'])
        chrom = str(alignment_attrs['mapped_chrom'])

    chrom_start = alignment_attrs['mapped_start']
    return strand, alignstrand, chrom, chrom_start


def _get_readid_from_fast5(h5file):
    """
    get read id from a fast5 file h5py obj
    :param h5file:
    :return: read id/name
    """
    first_read = list(h5file[reads_group].keys())[0]
    if sys.version_info[0] >= 3:
        try:
            read_id = str(h5file['/'.join([reads_group, first_read])].attrs['read_id'], 'utf-8')
        except TypeError:
            read_id = str(h5file['/'.join([reads_group, first_read])].attrs['read_id'])
    else:
        read_id = str(h5file['/'.join([reads_group, first_read])].attrs['read_id'])
    # print(read_id)
    return read_id


def _get_alignment_info_from_fast5(fast5_path, corrected_group, basecall_subgroup):
    """
    get alignment info (readid, strand (t/c), align strand, chrom, start) from fast5 files
    :param fast5_path:
    :param corrected_group:
    :param basecall_subgroup:
    :return: alignment information
    """
    try:
        h5file = h5py.File(fast5_path, 'r')
        corrgroup_path = '/'.join(['Analyses', corrected_group])
        if '/'.join([corrgroup_path, basecall_subgroup, 'Alignment']) in h5file:
            readname = _get_readid_from_fast5(h5file)
            strand, alignstrand, chrom, chrom_start = _get_alignment_attrs_of_each_strand(
                '/'.join([corrgroup_path, basecall_subgroup]), h5file)
            h5file.close()
            return readname, strand, alignstrand, chrom, chrom_start
        else:
            return '', '', '', '', ''
    except IOError:
        raise IOError('Error opening file. Likely a corrupted file.')
        return '', '', '', '', ''


def _normalize_signals(signals, normalize_method="mad"):
    if normalize_method == 'zscore':
        sshift, sscale = np.mean(signals), np.float_(np.std(signals))
    elif normalize_method == 'mad':

        sshift, sscale = np.median(signals), np.float_(robust.mad(signals))
    else:
        raise ValueError("")
    if sscale == 0.0:
        norm_signals = signals
    else:
        norm_signals = (signals - sshift) / sscale
    return np.around(norm_signals, decimals=6)


def _rescale_signals(rawsignals, scaling, offset):
    return np.array(scaling * (rawsignals + offset), dtype=np.float_)


def _get_scaling_of_a_read(fast5fp):
    global_key = "UniqueGlobalKey/"
    try:
        h5file = h5py.File(fast5fp, mode='r')
        channel_info = dict(list(h5file[global_key + 'channel_id'].attrs.items()))
        digi = channel_info['digitisation']
        parange = channel_info['range']
        offset = channel_info['offset']
        scaling = parange / digi

        h5file.close()
        # print(scaling, offset)
        return scaling, offset
    except IOError:
        print("the {} can't be opened".format(fast5fp))
        return None, None

def get_refloc_of_methysite_in_motif(seqstr, motifset):
    """
    :param seqstr:
    :param motifset:
    :param methyloc_in_motif: 0-based
    :return:
    """
    motifset = set(motifset)
    strlen = len(seqstr)
    motiflen = len(list(motifset)[0])
    methyloc_in_motif = (motiflen-1)//2
    sites = []
    for i in range(0, strlen - motiflen + 1):
        if seqstr[i:i + motiflen] in motifset:
            # print(i+methyloc_in_motif)
            sites.append(i+methyloc_in_motif)
    return sites

def _get_signals_rect(signals_list, signals_len=16):
    """
    signal features in matrix format
    :param signals_list:
    :param signals_len:
    :return: matrix format of signals
    """
    signals_rect = []
    for signals_tmp in signals_list:
        signals = list(np.around(signals_tmp, decimals=6)) # Evenly round to the given number of decimals.
        if len(signals) < signals_len:
            pad0_len = signals_len - len(signals)
            pad0_left = pad0_len // 2
            pad0_right = pad0_len - pad0_left
            signals = [0.] * pad0_left + signals + [0.] * pad0_right
        elif len(signals) > signals_len:
            signals = [signals[x] for x in sorted(random.sample(range(len(signals)),
                                                                signals_len))]
        signals_rect.append(signals)
    return signals_rect


def _get_insertions(ary, ins, ins_q, aln_mem):
    last_k = aln_mem[1],aln_mem[2]
    next_k = (ary[2], last_k[1] + 1)

    ins_k_up = (ary[0], ary[2], last_k[1])
    ins_k_down = (ary[0], ary[2], last_k[1] + 1)
    if not (ins_k_down) in ins_q:
        ins[next_k] = ins.get(next_k,0) + 1
        ins_q[ins_k_down].append(ord(ary[-4]) - 33)
    if not (ins_k_up) in ins_q:
        ins[last_k] = ins.get(last_k,0) + 1
        ins_q[ins_k_up].append(ord(ary[-4]) - 33)

    return ins, ins_q


def _get_deletions(ary, aln_mem, base, dele):
    k = (ary[2], int(ary[-3]))
    aln_mem = (ary[0],ary[2],int(ary[-3]))
    base[k] = ary[-2].upper()
    dele[k] = dele.get(k,0) + 1
    return dele, base, aln_mem


def _get_match_mismatch(ary, mis, mat, qual, base):
    ary[4] = ary[4].upper() # in case soft masked
    ary[7] = ary[7].upper()
    k = (ary[2], int (ary[-3]))
    aln_mem = (ary[0],ary[2],int(ary[-3]))
    qual[k] = ord(ary[-4]) - 33
    base[k] = ary[-2].upper()
    if (ary[-2] != ary[4]):
        mis[k] += 1
    else:
        mat[k] += 1
    return mis, mat, qual, base, aln_mem

def _get_slice_chunks(l, n):
    for i in range(0, len(l) - n):
        yield l[i:i + n]

def init_params():
    return defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), \
           defaultdict(int), defaultdict(list), {}

def get_feature_set(lines):
    qual, mis, mat, ins, dele, ins_q, base = init_params()
    for ary in lines:
        if ary[-1] == 'M':
            mis, mat, qual, base, aln_mem = _get_match_mismatch(
                ary, mis, mat, qual, base)

        if ary[-1] == 'D':
            dele, base, aln_mem = _get_deletions(ary, aln_mem, base, dele)

        if ary[-1] == 'I':
            ins, ins_q = _get_insertions(ary, ins, ins_q, aln_mem)

    return arrange_features(qual, mis, mat, ins, dele, base)

def arrange_features(qual, mis, mat, ins, dele, base):
    lines = []
    for k in base.keys():
        Mis = mis[k]; Mat = mat[k]
        Del = dele[k]; Ins = ins[k]; q_lst = qual[k]
        lines.append([k[0], k[1], base[k], q_lst, Mat, Mis, Ins, Del])
    return lines


def get_kmer_set(features, kmer_len, motif_seqs):
    position = _get_slice_chunks([item[-7] for item in features], kmer_len)
    sequence = _get_slice_chunks([item[-6] for item in features], kmer_len)
    quality = _get_slice_chunks([item[-5] for item in features], kmer_len)
    mismatch = _get_slice_chunks([item[-3] for item in features], kmer_len)
    insertion = _get_slice_chunks([item[-2] for item in features], kmer_len)
    deletion = _get_slice_chunks([item[-1] for item in features], kmer_len)

    loc = int(np.floor(kmer_len / 2))
    motifset = set(motif_seqs)
    motiflen = len(list(motifset)[0])
    mod_loc = (motiflen-1)//2

    lines = []
    for pos, seq, qual, mis, ins, dele in zip(
            position, sequence, quality, mismatch, insertion, deletion):
        if ''.join(seq[loc - mod_loc: loc + motiflen - mod_loc]) in motifset:

            pos = pos[loc]; seq = ''.join(seq)
            lines.append([pos, features[0][1], seq, qual, mis, ins, dele])
    return lines
    '''
    num_bases = (kmer_len - 1) // 2
    motifset = set(motif_seqs)
    motiflen = len(list(motifset)[0])
    methyloc_in_motif = (motiflen-1)//2

    errors_features_in_motif = []
    for pos, seq, qual, mis, ins, dele in zip(
            position, sequence, quality, mismatch, insertion, deletion):
        if ''.join(seq[num_bases - methyloc_in_motif: num_bases + motiflen - methyloc_in_motif]) in motif_seqs:
            pos = pos[num_bases]
            seq = ''.join(seq)
            errors_features_in_motif.append([pos, errors_features[0][1], seq, qual, mis, ins, dele])

    return errors_features_in_motif
    '''


def _get_basecall_errors(errors_dir, readname, kmer_len, motif_seqs):
    if not errors_dir.endswith('/'):
        errors_dir += '/'
    read_info = errors_dir + readname + '.txt'
    lines = []
    with open(read_info,"rt") as fh:
        for l in fh:
            if l.startswith('#'):
                continue
            lines.append(l.strip().split())

    features = get_feature_set(lines)
    #print(features)
    return get_kmer_set(features, kmer_len, motif_seqs)

def _extract_features(fast5s, corrected_group, basecall_subgroup, normalize_method, chrom2len, motif_seqs, kmer_len, signals_len, errors_dir):
    features_list = []
    error = 0
    for fast5_path in fast5s:
        try:
            readname, strand, alignstrand, chrom, chrom_start = _get_alignment_info_from_fast5(fast5_path, corrected_group, basecall_subgroup)
            raw_signal, events = _get_label_raw(fast5_path, corrected_group, basecall_subgroup)
            error_features = _get_basecall_errors(errors_dir, readname, kmer_len, motif_seqs)
            raw_signal = raw_signal[::-1]

            scaling, offset = _get_scaling_of_a_read(fast5_path)
            if scaling is not None:
                raw_signal = _rescale_signals(raw_signal, scaling, offset)
            norm_signals = _normalize_signals(raw_signal, normalize_method)

            readseq, signal_list = "", []
            for e in events: # events = list(zip(starts, lengths, base))
                readseq += str(e[2])
                signal_list.append(norm_signals[e[0]:(e[0] + e[1])])

            # strand transform
            chromlen = chrom2len[chrom]
            if alignstrand == '+':
                chrom_start_in_alignstrand = chrom_start
            else:
                chrom_start_in_alignstrand = chromlen - (chrom_start + len(readseq))
            # get motif_seq site
            tsite_locs = get_refloc_of_methysite_in_motif(readseq, set(motif_seqs))

            num_bases = (kmer_len - 1) // 2
            for loc_in_read in tsite_locs:
                if num_bases <= loc_in_read < len(readseq) - num_bases: # get kmer features
                    loc_in_ref = loc_in_read + chrom_start_in_alignstrand
                    if alignstrand == '-':
                        pos = chromlen - 1 - loc_in_ref # 还是要返回去
                        pos_error = [item[0] for item in error_features]
                    else:
                        pos = loc_in_ref
                        pos_error = [item[0]-1 for item in error_features]
                    k_mer = readseq[(loc_in_read - num_bases):(loc_in_read + num_bases + 1)]
                    k_signals = signal_list[(loc_in_read - num_bases):(loc_in_read + num_bases + 1)]
                    signal_lens = [len(x) for x in k_signals]
                    signal_means = [np.mean(x) for x in k_signals]
                    signal_median = [np.median(x) for x in k_signals]
                    signal_stds = [np.std(x) for x in k_signals]
                    k_signals_rect = _get_signals_rect(k_signals, signals_len)
                    comb_err = error_features[np.argwhere(np.asarray(pos_error) == pos)[0][0]]
                    qual, mis, ins, dele = comb_err[-4], comb_err[-3], comb_err[-2], comb_err[-1]

                    ####################
                    # print([chrom, pos, alignstrand, loc_in_ref, readname, strand,
                    #                       k_mer, signal_means, signal_median, signal_stds, signal_lens,
                    #                       k_signals_rect])
                    features_list.append([chrom, pos, alignstrand, loc_in_ref, readname, strand,
                                          k_mer, signal_means, signal_median, signal_stds, signal_lens,
                                          k_signals_rect, qual, mis, ins, dele])

        except FileNotFoundError as fnfe:
            error +=1
            continue
        except KeyError as ke:
            error +=1
            continue
        except RuntimeError as re:
            error +=1
            continue

    return features_list, error

def _feature2str(features):
    chrom, pos, alignstrand, loc_in_ref, readname, strand, k_mer, signal_means, signal_median, signal_stds, signal_lens, k_signals_rect, qual, mis, ins, dele = features
    means_text = ','.join([str(x) for x in np.around(signal_means, decimals=6)])
    median_test = ','.join([str(x) for x in np.around(signal_median, decimals=6)])
    stds_text = ','.join([str(x) for x in np.around(signal_stds, decimals=6)])
    signal_len_text = ','.join([str(x) for x in signal_lens])
    k_signals_text = ';'.join([",".join([str(y) for y in x]) for x in k_signals_rect])
    qual_text = ','.join([str(x) for x in qual])
    mis_text = ','.join([str(x) for x in mis])
    ins_text = ','.join([str(x) for x in ins])
    dele_text = ','.join([str(x) for x in dele])

    return "\t".join([chrom, str(pos), alignstrand, str(loc_in_ref), readname, strand, k_mer, means_text, median_test, stds_text, signal_len_text, k_signals_text, qual_text, mis_text, ins_text, dele_text])



def writefile(outfile_dir):
    if os.path.exists(outfile_dir):
        if os.path.isfile(outfile_dir):
            raise FileExistsError("{} already exists as a file, please use another write_dir".format(outfile_dir))
    else:
        os.makedirs(outfile_dir)
    features_out = []
    wf = open("/".join([outfile_dir, "features.txt"]), "w+")
    return wf

'''
def _extract_preprocess(fast5_dir)
    fast5_files = get_fast5_files(fast5_dir)
    # print("{} fast5 files in total..".format(len(fast5_files)))
    pbar = tqdm(total=len(fast5_files))
    pbar.set_description('extract features process')
    update = lambda *args: pbar.update()

    if n_process > 1:
        pool = mp.Pool(processes=n_process)
    else:
        pool = itertools
    # args = [(fn, corrected_group, basecall_subgroup, normalize_method, chrom2len, motif_seqs, kmer_len, signals_len) for fn in fast5_files]
    wf = writefile(outfile_dir)
    for fn in fast5_files:
        args = [fn, corrected_group, basecall_subgroup, normalize_method, chrom2len, motif_seqs, kmer_len, signals_len, errors_dir]
        result = pool.apply_async(extract_fast5_features, args, callback=update)

        feature_output_process(result.get(), wf)
    # results = list(pool.starmap(extract_fast5_features, args))
    # feature_output(results, outfile_dir)
    wf.close()
    print("finished!")
'''
def _fill_files_queue(fast5s_q, fast5_files, batch_size):
    for i in np.arange(0, len(fast5_files), batch_size):
        fast5s_q.put(fast5_files[i:(i+batch_size)])
    return

def get_a_batch_features_str(fast5s_q, featurestr_q, errornum_q,
                             corrected_group, basecall_subgroup, normalize_method,
                             motif_seqs, chrom2len, kmer_len, signals_len, errors_dir):

    f5_num = 0
    while True:
        if fast5s_q.empty():
            time.sleep(time_wait)
        fast5s = fast5s_q.get()
        if fast5s == "kill":
            fast5s_q.put("kill")
            break
        f5_num += len(fast5s)

        features_list, error_num = _extract_features(fast5s, corrected_group, basecall_subgroup,
                                                     normalize_method, chrom2len, motif_seqs,
                                                     kmer_len, signals_len, errors_dir)
        features_str = []
        for features in features_list:
            features_str.append(_feature2str(features))

        errornum_q.put(error_num)
        featurestr_q.put(features_str)
        while featurestr_q.qsize() > queen_size_border:
            time.sleep(time_wait)
    print("extrac_features process-{} ending, proceed {} fast5s".format(os.getpid(), f5_num))

def _write_featurestr_to_file(write_fp, featurestr_q):
    with open(write_fp, 'w') as wf:
        while True:
            # during test, it's ok without the sleep(time_wait)
            if featurestr_q.empty():
                time.sleep(time_wait)
                continue
            features_str = featurestr_q.get()
            if features_str == "kill":
                print('write_process-{} finished'.format(os.getpid()))
                break
            for one_features_str in features_str:
                wf.write(one_features_str + "\n")
            wf.flush()


def _write_featurestr_to_dir(write_dir, featurestr_q, w_batch_num):
    if os.path.exists(write_dir):
        if os.path.isfile(write_dir):
            raise FileExistsError("{} already exists as a file, please use another write_dir".format(write_dir))
    else:
        os.makedirs(write_dir)

    file_count = 0
    wf = open("/".join([write_dir, str(file_count) + ".tsv"]), "a+")
    batch_count = 0
    while True:
        # during test, it's ok without the sleep(time_wait)
        if featurestr_q.empty():
            time.sleep(time_wait)
            continue
        features_str = featurestr_q.get()
        if features_str == "kill":
            print('write_process-{} finished'.format(os.getpid()))
            break

        if batch_count >= w_batch_num:
            wf.flush()
            wf.close()
            file_count += 1
            wf = open("/".join([write_dir, str(file_count) + ".tsv"]), "a+")
            batch_count = 0
        for one_features_str in features_str:
            # fcntl.flock(wf.fileno(), fcntl.LOCK_EX) #file write lock
            wf.write(one_features_str + "\n")
            # fcntl.flock(wf,fcntl.LOCK_UN)
        batch_count += 1


def _write_featurestr(write_fp, featurestr_q, w_batch_num=10000, is_dir=False):
    if is_dir:
        _write_featurestr_to_dir(write_fp, featurestr_q, w_batch_num)
    else:
        _write_featurestr_to_file(write_fp, featurestr_q)


def _extract_preprocess(fast5_dir, is_recursive, motifs, bedfile, f5_batch_num):

    fast5_files = get_fast5_files(fast5_dir, is_recursive)
    print("{} fast5 files in total..".format(len(fast5_files)))

    print("parse the motifs string..")
    motif_seqs = get_motifs(motifs)

    print("read genome reference file..")
    chrom2len = get_chrom2len_from_bed(bedfile) #gencode.v28.transcripts.sid.fa.bed
    # print("read position file: {}".format(position_file))
    # positions = None
    # if position_file is not None:
    #     positions = _read_position_file(position_file)
    #
    # regioninfo = parse_region_str(regionstr)
    # chrom, start, end = regioninfo
    # print("parse region of interest: {}, [{}, {})".format(chrom, start, end))

    # fast5s_q = mp.Queue()
    fast5s_q = Queue()
    _fill_files_queue(fast5s_q, fast5_files, f5_batch_num)

    return motif_seqs, chrom2len, fast5s_q, len(fast5_files)



def extract_process(fast5_dir, n_process, is_recursive,  corrected_group, basecall_subgroup, normalize_method, bedfile, kmer_len, signals_len, errors_dir, write_path, w_is_dir, batch_size, w_batch_num):
    print("[main] extract_features starts..")
    start = time.time()
    motif_seqs, chrom2len, fast5s_q, len_fast5s= _extract_preprocess(fast5_dir, is_recursive,"DRACH", bedfile, batch_size)

    # featurestr_q = mp.Queue()
    # errornum_q = mp.Queue()
    featurestr_q = Queue()
    errornum_q = Queue()

    featurestr_procs = []
    if n_process > 1:
        n_process -= 1
    fast5s_q.put("kill")
    for _ in range(n_process):
        p = mp.Process(target=get_a_batch_features_str, args=(fast5s_q, featurestr_q, errornum_q,
                                                              corrected_group, basecall_subgroup,
                                                              normalize_method, motif_seqs,
                                                              chrom2len, kmer_len, signals_len, errors_dir))
        p.daemon = True
        p.start()
        featurestr_procs.append(p)

    # print("write_process started..")
    p_w = mp.Process(target=_write_featurestr, args=(write_path, featurestr_q, w_batch_num, w_is_dir))
    p_w.daemon = True
    p_w.start()

    errornum_sum = 0
    while True:
        # print("killing feature_p")
        running = any(p.is_alive() for p in featurestr_procs)
        while not errornum_q.empty():
            errornum_sum += errornum_q.get()
        if not running:
            break

    for p in featurestr_procs:
        p.join()

    # print("finishing the write_process..")
    featurestr_q.put("kill")

    p_w.join()

    print("%d of %d fast5 files failed..\n"
          "[main] extract_features costs %.1f seconds.." % (errornum_sum, len_fast5s,
                                                            time.time() - start))

    return

def check(kmer_len):
    if kmer_len % 2 == 0:
        raise ValueError("kmer_len must be odd")

def get_chrom2len_from_bed(bedfilename):
    beddata = pd.read_csv(bedfilename, sep='\t')
    dict1 = dict(zip(beddata['ID'], beddata['length']))
    return dict1

def str2bool(v):
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")

def main():
    parser = argparse.ArgumentParser(description='extract features from corrected(tombo) fast5')
    parser.add_argument("-i", "--fast5_dir", required=True, help="the directory of fast5 files")
    parser.add_argument("--recursively", action="store", type=str, required=False,
                          default='yes',
                          help='is to find fast5 files from fast5_dir recursively. '
                               'default true, t, yes, 1')
    parser.add_argument("-r", "--reference_path", required=False, help="the reference file to be used: *.fa")
    parser.add_argument("-o", "--write_path", required=False, help="the directory of features outfile")
    parser.add_argument("--w_is_dir", action="store",
                           type=str, required=False, default="no",
                           help='if using a dir to save features into multiple files')
    parser.add_argument("--corrected_group", required=False, default='RawGenomeCorrected_001',
                        help='the corrected_group of fast5 files after ''tombo re-squiggle. default RawGenomeCorrected_001')
    parser.add_argument("--basecall_subgroup", required=False, default='BaseCalled_template',
                        help='the corrected subgroup of fast5 files. default BaseCalled_template')
    parser.add_argument("-n", "--n_process", type=int, required=True, default="2")
    parser.add_argument("--normalize_method", required=False, default="mad", choices=["mad", "zscore"],
                        help="the way for normalizing signals in read level. mad or zscore, default mad")
    parser.add_argument("--errors_dir", required=True, help="errors tsv file")
    parser.add_argument("-k", "--kmer_len", type=int, required=True, help="errors tsv file")
    parser.add_argument("-s", "--signals_len", type=int, required=True, help="errors tsv file")
    parser.add_argument("-b", "--bedfile", required=True, help="fasta.bed filename")
    parser.add_argument("--f5_batch_size", action="store", type=int, default=20, required=False,
                        help="number of files to be processed by each process one time, default 20")
    parser.add_argument("--w_batch_num", action="store",type=int, default=200,required=False,
                        help='features batch num to save in a single writed file when --is_dir is true')

    args = parser.parse_args()

    fast5_dir = args.fast5_dir
    n_process = args.n_process
    is_recursive = str2bool(args.recursively)
    corrected_group = args.corrected_group
    basecall_subgroup = args.basecall_subgroup
    normalize_method = args.normalize_method
    #motif_seqs = ['AGACA', 'AGACC', 'AGACT', 'AAACA', 'AAACC', 'AAACT', 'GGACA', 'GGACC', 'GGACT', 'GAACA', 'GAACC', 'GAACT', 'TGACA', 'TGACC', 'TGACT', 'TAACA', 'TAACC', 'TAACT']

    kmer_len = args.kmer_len
    signals_len = args.signals_len
    bedfile = args.bedfile

    errors_dir = args.errors_dir
    write_path = args.write_path
    w_is_dir = str2bool(args.w_is_dir)
    f5_batch_size = args.f5_batch_size
    w_batch_num = args.w_batch_num
#################################################################### ENST00000511472.5
    # chrom2len = {"ENST00000262193.6":928, "ENST00000595831.5":1797, "ENST00000511472.5":369, "ENST00000483945.1":177, 'ENST00000253023.7':1508 }
    #chrom2len = {"cc6m_2709_T7_ecorv":2742, "cc6m_2459_T7_ecorv":2492, "cc6m_2595_T7_ecorv":2625, "cc6m_2244_T7_ecorv":2276}

    check(kmer_len)

    extract_process(fast5_dir, n_process, is_recursive, corrected_group, basecall_subgroup, normalize_method, bedfile, kmer_len, signals_len, errors_dir, write_path, w_is_dir, f5_batch_size, w_batch_num)

if __name__ == '__main__':
    sys.exit(main())

# /home/nipeng/data/xujr/test_eval_data/fast5_guppy/workspace/GISPC936_20181120_FAK27249_MN18749_sequencing_run_SHO_20112018_EmptyE2_9_36177_read_540_ch_214_strand.fast5