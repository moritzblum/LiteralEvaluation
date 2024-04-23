import argparse
import collections
import numpy as np

"""
This script creates relational ablations of existing datastes, meaning that it creates new datasets by 
removing certain relational relations from the original dataset.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FB15k-237', help='Dataset to create relational ablations from.')
    DATASET = parser.parse_args().dataset

    uri_2_freq = []
    triples = []
    with open(f'./data/{DATASET}/train.txt') as train_in:
        for line in train_in:
            triple = line.rstrip().split('\t')
            uri_2_freq.extend(triple)
            triples.append(triple)
    uri_2_freq = collections.Counter(uri_2_freq)
    num_triples = len(triples)

    for del_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        del_frac_label = str(int(del_frac * 100))
        del_num = int(del_frac * num_triples)

        permuted_triples = [triples[i] for i in np.random.permutation(np.arange(0, num_triples))]

        triples_maintained = []
        for head, relation, tail in permuted_triples:

            if del_num >= 1 and uri_2_freq[head] > 1 and uri_2_freq[relation] > 1 and uri_2_freq[tail] > 1:
                del_num -= 1
                continue
            else:
                triples_maintained.append([head, relation, tail])

        with open(f'./data/{DATASET}/train_pruned_{del_frac_label}.txt', 'w') as pruned_out:
            for triple in triples_maintained:
                pruned_out.write('\t'.join(triple) + '\n')
