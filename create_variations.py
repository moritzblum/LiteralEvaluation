import argparse

from torch.utils.data import Dataset
import os.path as osp
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


class LiteralLinkPredDataset(Dataset):
    # Only consider numerical literal features, the rest is commented out
    def __getitem__(self, index):
        # placeholder
        return None

    def __init__(self, triple_file, transform=None, target_transform=None):
        # Parameters
        self.triple_file = triple_file
        self.transform = transform
        self.target_transform = target_transform
        self.df_literals_num = pd.read_csv(osp.join(self.triple_file, 'literals/numerical_literals.txt' if DATASET_NAME != 'LitWD48K' else 'literals/numerical_literals_double.txt'), header=None, sep='\t')

        # Load external mappings of entities and relations to indices
        vocab_entities = np.load(osp.join(self.triple_file, "vocab_e1"), allow_pickle=True)
        self.entity2id = vocab_entities[0]

        self.relation2id = vocab_entities[1]

        # Literal data
        self.literals_num, self.attr_relations_num = self.load_literals_and_attr_relations_num()

    def load_literals_and_attr_relations_num(self):
        # with E = number of embeddings, R = number of attributive relations, V = feature dim
        print('Start loading numerical literals: E x R')
        attr_relations_num_unique = list(self.df_literals_num[1].unique())

        print("Unique numerical attributive relations: ", len(attr_relations_num_unique))

        attr_relation_num_2_id = {attr_relations_num_unique[i]: i for i in range(len(attr_relations_num_unique))}

        # Map entities to ids
        # Drop all literals that have entities that are not in the training set
        self.df_literals_num = self.df_literals_num[self.df_literals_num[0].isin(self.entity2id.keys())]
        self.df_literals_num[0] = self.df_literals_num[0].map(self.entity2id).astype(int)
        # Map attributive relations to ids
        self.df_literals_num[1] = self.df_literals_num[1].map(attr_relation_num_2_id).astype(int)
        # Change literal values to float
        self.df_literals_num[2] = self.df_literals_num[2].astype(float)

        # Extract numerical literal feature vectors for each entity for literal values and attributive relations
        features_num = []
        features_num_attr = []
        for i in tqdm(range(len(self.entity2id) + 2)):
            df_i = self.df_literals_num[self.df_literals_num[0] == i]

            feature_i = torch.zeros(len(attr_relations_num_unique))
            feature_i_attr = torch.zeros(len(attr_relations_num_unique))
            for index, row in df_i.iterrows():
                # Numerical literal values: row[1] = attributive relation index, row[2] = literal value as float
                feature_i[int(row[1])] = float(row[2])

                # One-hot encoding for attributive relations
                feature_i_attr[int(row[1])] = 1

            features_num.append(feature_i)
            features_num_attr.append(feature_i_attr)
        features_num = torch.stack(features_num)
        features_num_attr = torch.stack(features_num_attr)

        # Normalize numerical literals and attributive relations
        max_lit, min_lit = torch.max(features_num, dim=0).values, torch.min(features_num, dim=0).values
        features_num = (features_num - min_lit) / (max_lit - min_lit + 1e-8)
        features_num_attr -= features_num_attr.mean(dim=0, keepdim=True)

        return features_num, features_num_attr

    def filter_literals_by_attr_relation_frequency(self, threshold=100):
        # Only filters numerical literals

        print(f"Filtering literals by attributive relation frequency with threshold {threshold}...")
        # Count occurences of attributive relations
        attr_relations_num_counts = self.df_literals_num[1].value_counts()
        # Filter attributive relations by frequency
        attr_relations_num_filtered = attr_relations_num_counts[attr_relations_num_counts > threshold].index

        print(f"Literals num before: {len(attr_relations_num_counts)} after: {len(attr_relations_num_filtered)}")

        # Filter literals_num (literals_txt only has one description literal per entity -> no filtering needed)
        self.literals_num = self.literals_num[:, attr_relations_num_filtered]

        # Filter attributive relations
        self.attr_relations_num = self.attr_relations_num[:, attr_relations_num_filtered]


if __name__ == '__main__':
    # For this script to work, the numerical literal file has to be "numerical_literals.txt" and be stored in the
    # data/{dataset_name}/literals directory.

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FB15k-237')
    args = parser.parse_args()
    DATASET_NAME = args.dataset
    print(f'Dataset: {DATASET_NAME}')

    if not osp.isfile(f'data/{DATASET_NAME}/literals/processed.pt'):
        print('Process dataset...')
        dataset = LiteralLinkPredDataset(f'data/{DATASET_NAME}')
        torch.save(dataset, f'data/{DATASET_NAME}/literals/processed.pt')

    dataset = torch.load(f'data/{DATASET_NAME}/literals/processed.pt')

    # Save numerical literals to numpy file
    np.save(f'data/{DATASET_NAME}/literals/numerical_literals_rep.npy', dataset.literals_num.numpy())

    # Save attributive relation type literals to numpy file
    np.save(f'data/{DATASET_NAME}/literals/numerical_literals_attr.npy', dataset.attr_relations_num.numpy())

    # Save filtered numerical literals to numpy file
    filtered_ds_100 = dataset
    filtered_ds_100.filter_literals_by_attr_relation_frequency(100)
    np.save(f'data/{DATASET_NAME}/literals/numerical_literals_filtered_100.npy',
            filtered_ds_100.literals_num.numpy())
