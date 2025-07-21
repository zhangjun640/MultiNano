import os
import fnmatch
import random
import numpy as np
import gc


def get_fast5_files(fast5_dir, is_recursive=True):
    fast5_dir = os.path.abspath(fast5_dir)  # get abs directoty
    fast5_files = []
    if is_recursive:
        for root, dirnames, filenames in os.walk(fast5_dir):
            for filename in fnmatch.filter(filenames, "*hdf5"):
                fast5_path = os.path.join(root, filename)
                fast5_files.append(fast5_path)
    else:
        for filename in os.listdir(fast5_dir):
            if filename.endswith(".hdf5"):
                fast5_path = '/'.join([fast5_dir, filename])
                fast5_files.append(fast5_path)
    return fast5_files


def split_train_val(file_list):
    val_list = sorted(file_list, key=lambda _: random.random())
    length = int(0.75 * len(file_list))
    return val_list[:length], val_list[length:]


def sample(dataset_dir):
    # 对HCT的训练数据进行采样
    mod_file_list = get_fast5_files(os.path.join(dataset_dir, "mod"))
    # 随机选择同样数目的unm
    unm_file_list = random.sample(get_fast5_files(os.path.join(dataset_dir, "unm")), len(mod_file_list))

    # 按照3:1划分训练和验证集
    train_mod, val_mod = split_train_val(mod_file_list)
    train_unm, val_unm = split_train_val(unm_file_list)

    train_mod.extend(train_unm)
    val_mod.extend(val_unm)
    train_mod = sorted(train_mod, key=lambda _: random.random())
    val_mod = sorted(val_mod, key=lambda _: random.random())

    # random.shuffle(train_mod)
    # random.shuffle(val_mod)
    print("shuffle train: ", len(train_mod), " unm: ", len(val_mod))

    return train_mod, val_mod


# ================sampling txt feafure file =================================
def count_line_num(sl_filepath, fheader=False):
    count = 0
    with open(sl_filepath, 'r') as rf:
        if fheader:
            next(rf)
        for _ in rf:
            count += 1
    # print('done count the lines of file {}'.format(sl_filepath))
    return count


def read_one_shuffle_info(filepath, shuffle_lines_num, total_lines_num, checked_lines_num, isheader):
    with open(filepath, 'r') as rf:
        if isheader:
            next(rf)
        count = 0
        while count < checked_lines_num:
            next(rf)
            count += 1

        count = 0
        lines_info = []
        lines_num = min(shuffle_lines_num, (total_lines_num - checked_lines_num))
        for line in rf:
            if count < lines_num:
                lines_info.append(line.strip())
                count += 1
            else:
                break
        print('done reading file {}'.format(filepath))
        return lines_info


def shuffle_samples(samples_info):
    mark = list(range(len(samples_info)))
    np.random.shuffle(mark)
    shuffled_samples = []
    for i in mark:
        shuffled_samples.append(samples_info[i])
    return shuffled_samples


def write_to_one_file_append(features_info, wfilepath):
    with open(wfilepath, 'a') as wf:
        for i in range(0, len(features_info)):
            wf.write(features_info[i] + '\n')
    print('done writing features info to {}'.format(wfilepath))


def concat_two_files(file1, file2, concated_fp, shuffle_lines_num=2000000,
                     lines_num=1000000000000, isheader=False):
    open(concated_fp, 'w').close()

    if isheader:
        rf1 = open(file1, 'r')
        wf = open(concated_fp, 'a')
        wf.write(next(rf1))
        wf.close()
        rf1.close()

    f1line_count = count_line_num(file1, isheader)
    f2line_count = count_line_num(file2, False)

    line_ratio = float(f2line_count) / f1line_count
    shuffle_lines_num2 = round(line_ratio * shuffle_lines_num) + 1

    checked_lines_num1, checked_lines_num2 = 0, 0
    while checked_lines_num1 < lines_num or checked_lines_num2 < lines_num:
        file1_info = read_one_shuffle_info(file1, shuffle_lines_num, lines_num, checked_lines_num1, isheader)
        checked_lines_num1 += len(file1_info)
        file2_info = read_one_shuffle_info(file2, shuffle_lines_num2, lines_num, checked_lines_num2, False)
        checked_lines_num2 += len(file2_info)
        if len(file1_info) == 0 and len(file2_info) == 0:
            break
        samples_info = shuffle_samples(file1_info + file2_info)
        write_to_one_file_append(samples_info, concated_fp)

        del file1_info
        del file2_info
        del samples_info
        gc.collect()
    print('done concating files to: {}'.format(concated_fp))


def random_select_file_rows(ori_file, w_file, w_other_file=None, maxrownum=100000000, header=False):
    """

    :param ori_file:
    :param w_file:
    :param w_other_file:
    :param maxrownum:
    :param header:
    :return:
    """
    # whole_rows = open(ori_file).readlines()
    # nrows = len(whole_rows) - 1

    nrows = 0
    with open(ori_file) as rf:
        for _ in rf:
            nrows += 1
    if header:
        nrows -= 1
    print('thera are {} lines (rm header if a header exists) in the file {}'.format(nrows, ori_file))

    actual_nline = maxrownum
    if nrows <= actual_nline:
        actual_nline = nrows
        print('gonna return all lines in ori_file {}'.format(ori_file))

    random_lines = random.sample(range(1, nrows+1), actual_nline)
    random_lines = [0] + sorted(random_lines)
    random_lines[-1] = nrows

    wf = open(w_file, 'w')
    if w_other_file is not None:
        wlf = open(w_other_file, 'w')
    with open(ori_file) as rf:
        if header:
            lineheader = next(rf)
            wf.write(lineheader)
            if w_other_file is not None:
                wlf.write(lineheader)
        for i in range(1, len(random_lines)):
            chosen_line = ''
            for j in range(0, random_lines[i]-random_lines[i-1] - 1):
                other_line = next(rf)
                if w_other_file is not None:
                    wlf.write(other_line)
            chosen_line = next(rf)
            wf.write(chosen_line)
    wf.close()
    if w_other_file is not None:
        wlf.close()
    print('random_select_file_rows finished..')


def sample_txt(pos_txt, neg_txt, tratio=0.75):
    line_cnt = count_line_num(pos_txt, False)
    line_cnt_t = int(round(line_cnt * tratio))
    line_cnt_v = line_cnt - line_cnt_t
    fname, fext = os.path.splitext(pos_txt)
    pos_train = fname + ".train" + fext
    pos_valid = fname + ".valid" + fext
    random_select_file_rows(pos_txt, pos_train, pos_valid, line_cnt_t)

    fname, fext = os.path.splitext(neg_txt)
    neg_sampled = fname + ".sampled" + fext
    random_select_file_rows(neg_txt, neg_sampled, None, line_cnt)
    neg_train = fname + ".train" + fext
    neg_valid = fname + ".valid" + fext
    random_select_file_rows(neg_sampled, neg_train, neg_valid, line_cnt_t)

    fname, fext = os.path.splitext(pos_txt)
    train_file = fname + ".bneg.train" + fext
    valid_file = fname + ".bneg.valid" + fext
    concat_two_files(pos_train, neg_train, train_file)
    concat_two_files(pos_valid, neg_valid, valid_file)
    os.remove(pos_train)
    os.remove(pos_valid)
    os.remove(neg_train)
    os.remove(neg_valid)
    os.remove(neg_sampled)
    return train_file, valid_file

def sample_ara_txt(pos_txt, neg_txt, tratio=0.75):
    line_cnt = count_line_num(pos_txt, False)
    line_cnt_t = int(round(line_cnt * tratio))
    line_cnt_v = line_cnt - line_cnt_t
    fname, fext = os.path.splitext(pos_txt)
    pos_train = fname + ".train" + fext
    pos_valid = fname + ".valid" + fext
    random_select_file_rows(pos_txt, pos_train, pos_valid, line_cnt_t)

    fname, fext = os.path.splitext(neg_txt)
    # neg_sampled = fname + ".sampled" + fext
    # random_select_file_rows(neg_txt, neg_sampled, None, line_cnt)
    line_cnt = count_line_num(neg_txt, False)
    line_cnt_t = int(round(line_cnt * tratio))
    line_cnt_v = line_cnt - line_cnt_t
    neg_train = fname + ".train" + fext
    neg_valid = fname + ".valid" + fext
    random_select_file_rows(neg_txt, neg_train, neg_valid, line_cnt_t)

    fname, fext = os.path.splitext(pos_txt)
    train_file = fname + ".bneg.train" + fext
    valid_file = fname + ".bneg.valid" + fext
    concat_two_files(pos_train, neg_train, train_file)
    concat_two_files(pos_valid, neg_valid, valid_file)
    os.remove(pos_train)
    os.remove(pos_valid)
    os.remove(neg_train)
    os.remove(neg_valid)
    # os.remove(neg_sampled)
    return train_file, valid_file


def no_sample_txt(pos_txt, neg_txt, tratio=0.75):
    line_cnt = count_line_num(pos_txt, False)
    line_cnt_t = int(round(line_cnt * tratio))
    line_cnt_v = line_cnt - line_cnt_t
    fname, fext = os.path.splitext(pos_txt)
    pos_train = fname + ".train" + fext
    pos_valid = fname + ".valid" + fext
    random_select_file_rows(pos_txt, pos_train, pos_valid, line_cnt_t)

    
    line_cnt = count_line_num(neg_txt, False)
    line_cnt_t = int(round(line_cnt * tratio))
    # neg_sampled = fname + ".sampled" + fext
    # random_select_file_rows(neg_txt, neg_sampled, None, line_cnt)
    fname, fext = os.path.splitext(neg_txt)
    neg_train = fname + ".train" + fext
    neg_valid = fname + ".valid" + fext
    random_select_file_rows(neg_txt, neg_train, neg_valid, line_cnt_t)

    fname, fext = os.path.splitext(pos_txt)
    train_file = fname + ".bneg.train" + fext
    valid_file = fname + ".bneg.valid" + fext
    concat_two_files(pos_train, neg_train, train_file)
    concat_two_files(pos_valid, neg_valid, valid_file)
    os.remove(pos_train)
    os.remove(pos_valid)
    os.remove(neg_train)
    os.remove(neg_valid)
    # os.remove(neg_sampled)
	
    return train_file, valid_file

def sample_txt_xore(pos_txt, neg_txt, tratio=0.75):
    # line_cnt = count_line_num(pos_txt, False)
    # line_cnt_t = int(round(line_cnt * tratio))
    # line_cnt_v = line_cnt - line_cnt_t
    # fname, fext = os.path.splitext(pos_txt)
    # pos_train = fname + ".train" + fext
    # pos_valid = fname + ".valid" + fext
    # random_select_file_rows(pos_txt, pos_train, pos_valid, line_cnt_t)

    # fname, fext = os.path.splitext(neg_txt)
    # neg_sampled = fname + ".sampled" + fext
    # random_select_file_rows(neg_txt, neg_sampled, None, line_cnt)
    # neg_train = fname + ".train" + fext
    # neg_valid = fname + ".valid" + fext
    # random_select_file_rows(neg_sampled, neg_train, neg_valid, line_cnt_t)

    fname, fext = os.path.splitext(pos_txt)
    train_file = fname + ".bneg.train" + fext
    valid_file = fname + ".bneg.valid" + fext
    # concat_two_files(pos_train, neg_train, train_file)
    # concat_two_files(pos_valid, neg_valid, valid_file)
    # os.remove(pos_train)
    # os.remove(pos_valid)
    # os.remove(neg_train)
    # os.remove(neg_valid)
    # os.remove(neg_sampled)
	
    return train_file, valid_file