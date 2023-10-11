import json
from collections import Counter
import os.path as osp

import numpy as np

if __name__ == '__main__':

    entities = set()
    relations = set()
    triples = []

    #dataset_path = "./data/FB15k/"
    #dataset_path = "./datasets/FB15k-237/"
    #dataset_path = "./datasets/YAGO3-10/"
    dataset_path = "./datasets/WN18RR/"


    for split in ['train', 'valid', 'test']:
        with open(osp.join(dataset_path, f'{split}.txt')) as f:
            for line in f:
                head, relation, tail = line.strip().split('\t')
                triples.append([head, relation, tail])
                entities.add(head)
                entities.add(tail)
                relations.add(relation)

        rel_2_tuple = {r: [] for r in relations}
        for head, relation, tail in triples:
            rel_2_tuple[relation].append([head, tail])

    type_2_relations = {'1-1': [], '1-n': [], 'n-1': [], 'n-n': []}
    stats = []
    for relation in relations:
        nn = []
        for mode in ['head', 'tail']:
            ent_2_ent = {}
            for ent1, ent2 in rel_2_tuple[relation]:
                if mode == 'tail':
                    head, tail = ent1, ent2
                else:
                    tail, head = ent1, ent2
                if head not in ent_2_ent:
                    ent_2_ent[head] = []
                ent_2_ent[head].append(tail)
            if np.mean([len(v) for v in ent_2_ent.values()]) < 1.5:
                nn.append('1')
            else:
                nn.append('n')

        type_2_relations['-'.join(nn)].append(relation)
        stats.append('-'.join(nn))


    print(type_2_relations)
    json.dump(type_2_relations, open(osp.join(dataset_path, 'type_2_relations.json'), 'w'))

    for k, v in type_2_relations.items():
        print(k, len(v)/len(relations))






