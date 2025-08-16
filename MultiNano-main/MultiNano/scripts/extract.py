

import os, sys, time, random, argparse, multiprocessing as mp
from multiprocessing import Queue
import h5py, numpy as np
from statsmodels import robust
from tqdm import tqdm

# Environment configuration
os.environ["HDF5_PLUGIN_PATH"] = ""
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "2"

# Constants
reads_group       = "/Raw/Reads"
queen_size_border = 2000
time_wait         = 3

# utils
sys.path.append(os.getcwd())
from MultiNano.utils.data_utils import get_fast5_files, get_motifs

# Parse basecall error files
def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return x                    # Keep '.' as-is

def parse_line_fields_per_column(line):
    f = line.strip().split()
    if len(f) < 9:
        return None
    try:
        f[1] = _safe_int(f[1])
        f[3] = _safe_int(f[3])
        f[6] = _safe_int(f[6])      # Third-to-last column: read_index
    except Exception:
        return None
    try:
        f[4] = f[4][0].upper() if isinstance(f[4], str) else 'N'
        f[5] = f[5]
        f[7] = f[7][0].upper() if isinstance(f[7], str) else 'N'
        f[8] = f[8].upper()
    except Exception:
        return None
    return f

# Signal normalization / rescaling (unaltered)
def _normalize_signals(sig, method="mad"):
    if method == "zscore":
        mu, sd = np.mean(sig), float(np.std(sig))
    elif method == "mad":
        mu, sd = np.median(sig), float(robust.mad(sig))
    else:
        raise ValueError("Unknown normalize_method.")
    return np.around(sig if sd == 0 else (sig - mu) / sd, 6)

def _rescale_signals(raw, scl, off):
    return np.array(scl * (raw + off), dtype=np.float_)

def _get_scaling_of_a_read(fp):
    try:
        with h5py.File(fp) as h5:
            ch = dict(h5["UniqueGlobalKey/channel_id"].attrs.items())
            return ch["range"] / ch["digitisation"], ch["offset"]
    except Exception:
        return None, None

# fast5
def _get_label_raw(fp, corr_grp, subgrp):
    with h5py.File(fp) as h5:
        raw = list(h5[reads_group].values())[0]["Signal"][()]
        evt = h5[f"/Analyses/{corr_grp}/{subgrp}/Events"]
        s0  = dict(evt.attrs)["read_start_rel_to_raw"]
        starts = [s + s0 for s in evt["start"]]
        lens   = evt["length"].astype(np.int_)
        bases  = [b.decode() for b in evt["base"]]
    return raw, list(zip(starts, lens, bases))

def _get_readid_from_fast5(h5):
    first = list(h5[reads_group].keys())[0]
    rid   = h5[f"{reads_group}/{first}"].attrs["read_id"]
    return rid.decode() if isinstance(rid, bytes) else rid

def _get_alignment_info_from_fast5(fp, corr_grp, subgrp):
    with h5py.File(fp) as h5:
        aln = f"Analyses/{corr_grp}/{subgrp}/Alignment"
        if aln not in h5:
            return "", "", "", "", ""
        ai = h5[aln].attrs
        rid  = _get_readid_from_fast5(h5)
        strand = "t" if subgrp.endswith("template") else "c"
        aln_strand = ai["mapped_strand"].decode() if isinstance(ai["mapped_strand"], bytes) else ai["mapped_strand"]
        chrom      = ai["mapped_chrom"].decode()  if isinstance(ai["mapped_chrom"],  bytes) else ai["mapped_chrom"]
        chrom_start= ai["mapped_start"]
    return rid, strand, aln_strand, chrom, chrom_start

# Load basecall error files
def _get_basecall_errors(err_dir, rid):
    fp = os.path.join(err_dir, f"{rid}.txt")
    if not os.path.exists(fp):
        return [], None
    lines=[]
    with open(fp) as fh:
        for raw in fh:
            if raw.startswith("#") or not raw.strip():
                continue
            fld = parse_line_fields_per_column(raw)
            if fld: lines.append(fld)
    if not lines: return [], None

    try:                               
        first_idx = next(l[-3] for l in lines if isinstance(l[-3], int) and l[-2] != '.')
    except StopIteration:
        return [], None                 

    rel_pos = [(int(l[-3])-first_idx) if isinstance(l[-3], int) else '.' for l in lines]
    return rel_pos, lines

# Signal segmentation
def _get_signals_rect(sig_list, tgt_len=16):
    out=[]
    for s in sig_list:
        s=list(np.around(s,6))
        if len(s)<tgt_len:
            pad=tgt_len-len(s); out.append([0.]* (pad//2)+s+[0.]* (pad-pad//2))
        elif len(s)>tgt_len:
            idx=sorted(random.sample(range(len(s)),tgt_len)); out.append([s[i] for i in idx])
        else: out.append(s)
    return out

# Error mapping
def _build_event_index_to_error_line(err_lines):
    try:
        first_idx = next(l[-3] for l in err_lines if isinstance(l[-3], int) and l[-2] != '.')
    except StopIteration:
        return {}
    mapping={}
    for idx,l in enumerate(err_lines):
        if isinstance(l[-3],int):
            base_idx = l[-3] - first_idx
            mapping[base_idx]=(idx,l)
    return mapping

# Core feature extraction (mis/ins/del)
def _extract_features(f5s,corr_grp,subgrp,norm_m,
                      motif_seqs,klen,slen,err_dir):
    feats,err= [],0
    motifset=set(motif_seqs); mlen=len(next(iter(motifset))); center_off=(mlen-1)//2
    for fp in f5s:
        try:
            rid,strand,alnstrand,chrom,_ = _get_alignment_info_from_fast5(fp,corr_grp,subgrp)
            if not rid: err+=1; continue
            raw,events = _get_label_raw(fp,corr_grp,subgrp)
            _,err_lines= _get_basecall_errors(err_dir,rid)
            if not err_lines: err+=1; continue

            raw=raw[::-1]; scl,off=_get_scaling_of_a_read(fp)
            if scl is not None: raw=_rescale_signals(raw,scl,off)
            norm=_normalize_signals(raw,norm_m)

            bases=[e[2] for e in events]
            seg_sig=[norm[e[0]:e[0]+e[1]] for e in events]
            err_map=_build_event_index_to_error_line(err_lines)

            for i in range(len(bases)-mlen+1):
                if ''.join(bases[i:i+mlen]) not in motifset: continue
                center=i+center_off; s=max(0,center-klen//2); e=min(len(bases),center+klen//2+1)
                matches=[]; ok=True
                for bidx in range(s,e):
                    if bidx not in err_map: ok=False; break
                    eidx,eln=err_map[bidx]
                    if bases[bidx]!=eln[7]: ok=False; break
                    matches.append((bidx,eidx,eln))
                if not ok: continue

                k_sig=seg_sig[s:e]
                sig_means=[float(np.mean(x)) for x in k_sig]
                sig_meds =[float(np.median(x)) for x in k_sig]
                sig_stds =[float(np.std(x)) for x in k_sig]
                sig_lens =[len(x) for x in k_sig]

                quals,miss,ins,dele=[],[],[],[]
                for _,eidx,eln in matches:
                    qtype=eln[-1]
                    quals.append(0 if eln[5]=='.' else ord(eln[5])-33)
                    miss.append(1 if qtype=='S' else 0)
                    dele.append(1 if qtype=='D' else 0)
                    ins_flag=0
                    for off in (-1,1):
                        adj=eidx+off
                        if 0<=adj<len(err_lines) and err_lines[adj][-1]=='I': ins_flag=1; break
                    ins.append(ins_flag)

                feats.append([
                    chrom,center,alnstrand,center,rid,strand,
                    ''.join(bases[s:e]),
                    sig_means,sig_meds,sig_stds,sig_lens,
                    _get_signals_rect(k_sig,slen),
                    quals,miss,ins,dele
                ])
        except Exception as e:
            print(f"[DEBUG] Exception in _extract_features for {fp}: {e!r}"); err+=1
    return feats,err

# Feature serialization, file writing, multiprocessing scheduler, main


# Feature serialization
def _join(a): return ",".join(map(str,a))
def _join2(m): return ";".join(",".join(map(str,row)) for row in m)

def _feature2str(f):
    (chrom,pos,aln,loc,rid,strand,kmer,
     means,meds,stds,lens,rect,quals,miss,ins,dele) = f
    return "\t".join([
        chrom, str(pos), aln, str(loc), rid, strand, kmer,
        _join(np.around(means,6)), _join(np.around(meds,6)),
        _join(np.around(stds,6)),  _join(lens),
        _join2(rect),
        _join(quals), _join(miss), _join(ins), _join(dele)
    ])

# File writing functions
def _write_featurestr_to_file(outfile,q):
    with open(outfile,"w") as wf:
        while True:
            if q.empty(): time.sleep(time_wait); continue
            blk=q.get()
            if blk=="kill": break
            wf.write("\n".join(blk)+"\n")

def _write_featurestr_to_dir(outdir,q,batch_lim):
    os.makedirs(outdir,exist_ok=True)
    idx,cnt=0,0
    wf=open(f"{outdir}/{idx}.tsv","w")
    while True:
        if q.empty(): time.sleep(time_wait); continue
        blk=q.get()
        if blk=="kill": break
        if cnt>=batch_lim:
            wf.close(); idx+=1; wf=open(f"{outdir}/{idx}.tsv","w"); cnt=0
        wf.write("\n".join(blk)+"\n"); cnt+=1
    wf.close()

def _writer(entry,q,batch,is_dir):
    if is_dir: _write_featurestr_to_dir(entry,q,batch)
    else:      _write_featurestr_to_file(entry,q)
    print(f"[DEBUG] Writer {os.getpid()} done")

# Multiprocessing scheduler
def _fill(q, files, bs):
    for i in range(0,len(files),bs):
        q.put(files[i:i+bs])
    q.put("kill")

def _worker(f5_q,feat_q,err_q,
            corr_grp,subgrp,norm_m,motifs,klen,slen,err_dir):
    total=0
    while True:
        if f5_q.empty(): time.sleep(time_wait); continue
        batch=f5_q.get()
        if batch=="kill":
            f5_q.put("kill"); break
        total+=len(batch)
        feats,nerr=_extract_features(batch,corr_grp,subgrp,
                                     norm_m,motifs,klen,slen,err_dir)
        feat_q.put([_feature2str(f) for f in feats])
        err_q.put(nerr)
        while feat_q.qsize()>queen_size_border:
            time.sleep(time_wait)
    print(f"[DEBUG] Worker {os.getpid()} processed {total} fast5")

def extract_process(f5_dir,nproc,rec,
                    corr_grp,subgrp,norm_m,klen,slen,
                    err_dir,out_path,out_is_dir,
                    f5_bs,w_bs):
    t0=time.time()
    files=get_fast5_files(f5_dir,rec)
    motifs=get_motifs("DRACH")
    f5_q=Queue(); _fill(f5_q,files,f5_bs)
    feat_q=Queue(); err_q=Queue()

    workers=[]
    for _ in range(max(1,nproc-1)):
        p=mp.Process(target=_worker,
            args=(f5_q,feat_q,err_q,
                  corr_grp,subgrp,norm_m,motifs,klen,slen,err_dir))
        p.daemon=True; p.start(); workers.append(p)

    writer=mp.Process(target=_writer,
                      args=(out_path,feat_q,w_bs,out_is_dir))
    writer.daemon=True; writer.start()

    err_sum=0
    while any(p.is_alive() for p in workers):
        while not err_q.empty(): err_sum+=err_q.get()
        time.sleep(1)
    for p in workers: p.join()
    feat_q.put("kill"); writer.join()
    print(f"[DONE] {err_sum}/{len(files)} fast5 failed.  Time: {time.time()-t0:.1f}s")

# main
def str2bool(v): return v.lower() in ("yes","true","1","t")
def chk(k):
    if k%2==0: raise ValueError("kmer_len must be odd")

def main():
    ap=argparse.ArgumentParser("Extract features from Tombo-corrected fast5")
    ap.add_argument("-i","--fast5_dir",required=True)
    ap.add_argument("--recursively",default="yes")
    ap.add_argument("-o","--write_path",required=True)
    ap.add_argument("--w_is_dir",default="no")
    ap.add_argument("--corrected_group",default="RawGenomeCorrected_001")
    ap.add_argument("--basecall_subgroup",default="BaseCalled_template")
    ap.add_argument("-n","--n_process",type=int,default=2)
    ap.add_argument("--normalize_method",choices=["mad","zscore"],default="mad")
    ap.add_argument("--errors_dir",required=True)
    ap.add_argument("-k","--kmer_len",type=int,required=True)
    ap.add_argument("-s","--signals_len",type=int,required=True)
    ap.add_argument("--f5_batch_size",type=int,default=20)
    ap.add_argument("--w_batch_num",type=int,default=200)
    a=ap.parse_args(); chk(a.kmer_len)

    extract_process(
        f5_dir=a.fast5_dir, nproc=a.n_process,
        rec=str2bool(a.recursively),
        corr_grp=a.corrected_group, subgrp=a.basecall_subgroup,
        norm_m=a.normalize_method,
        klen=a.kmer_len, slen=a.signals_len,
        err_dir=a.errors_dir,
        out_path=a.write_path, out_is_dir=str2bool(a.w_is_dir),
        f5_bs=a.f5_batch_size, w_bs=a.w_batch_num
    )

if __name__=="__main__":
    sys.exit(main())
