import argparse
import gzip

key_sep = '||'


def aggre_features_to_site_level(input_file, output_file, mlabel="1"):
    last_site_key = ""
    feature_str = ""
    last_label = ""
    last_kmer = ""
    count = 0
    with open(output_file, 'w') as wf:
        if input_file.endswith(".gz"):
            infile = gzip.open(input_file, 'rt')
        else:
            infile = open(input_file, 'r')
        for line in infile:
            words = line.strip().split('\t')
            sampleid = "\t".join(words[0:3])
            site_key = key_sep.join([words[0], words[1]])
            kmer = words[6]
            features = words[7:16]
            if len(words) == 17:
                label = words[16]
            else:
                label = mlabel
            assert len(features) == 9
            if site_key != last_site_key:
                if last_site_key != "":
                    wf.write(last_sampleid + '\t' + str(count) + '\t' + last_kmer + '\t' + feature_str + '\t' + last_label +'\n')
                last_site_key = site_key
                feature_str = key_sep.join(features)
                count = 1
            else:
                if last_label != "":
                    assert last_label == label
                    assert last_kmer == kmer
                feature_str += '\t' + key_sep.join(features)
                count += 1
            last_label = label
            last_kmer = kmer
            last_sampleid = sampleid
        wf.write(last_sampleid + '\t' + str(count) + '\t' + last_kmer + '\t' + feature_str + '\t' + last_label + '\n')
        infile.close()


def main():
    parser = argparse.ArgumentParser(description='Aggregate features to site level')
    parser.add_argument('--input', type=str, help='Input file, sorted')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--mlabel', type=str, help='Missing label', default='1')
    args = parser.parse_args()

    aggre_features_to_site_level(args.input, args.output, args.mlabel)


if __name__ == '__main__':
    main()