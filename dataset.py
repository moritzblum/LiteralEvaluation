from sklearn.cluster import KMeans
from torch.utils.data import Dataset
import os.path as osp
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import spacy


class LiteralLinkPredDataset(Dataset):
    # Only consider numerical literal features, the rest is commented out
    def __getitem__(self, index):
        # placeholder
        return None

    def __init__(self, triple_file, transform=None, target_transform=None):
        # Parameters
        self.triple_file = triple_file
        self.embedding_dim = 300  # Mandated by Spacy
        self.transform = transform
        self.target_transform = target_transform

        # Data
        self.df_triples_train, self.df_triples_val, self.df_triples_test, self.df_literals_num, self.df_literals_txt \
            = self.load_dataframes()

        # Relational data
        # self.entities, self.relations = self.load_relational_data()

        #cself.num_entities = len(self.entities)
        # self.num_relations = len(self.relations)

        # Load external mappings of entities and relations to indices
        vocab_entities = np.load("vocab_e1", allow_pickle=True)
        self.entity2id = vocab_entities[0]

        print("Entity vocab lists")
        print(len(vocab_entities[0]), len(vocab_entities[1]), len(vocab_entities[2]), len(vocab_entities[3]))
        # Print entities that don't start with /m/
        print("Special entity tokens:")
        for entity in self.entity2id:
            if not entity.startswith('/m/'):
                print("->", entity)

        # Print number of unique entities starting with /m/
        print("Number of unique entities starting with /m/:",
              len(set([entity for entity in self.entity2id if entity.startswith('/m/')])))


        self.relation2id = vocab_entities[1]
        # self.relation2id = {relation: idx for idx, relation in enumerate(self.relations)}

        # self.edge_index_train, self.edge_index_val, self.edge_index_test = self.load_edge_indices()
        # self.edge_type_train, self.edge_type_val, self.edge_type_test = self.load_edge_types()

        # Add inverse relations to the data set
        # self.num_relations = self.num_relations * 2

        # print(f"Number of training triples: {len(self.df_triples_train)}")
        # print(f"Number of validation triples: {len(self.df_triples_val)}")
        # print(f"Number of test triples: {len(self.df_triples_test)}")
        # print(f"Edge index train shape: {self.edge_index_train.shape}")
        # print(f"Edge type train shape: {self.edge_type_train.shape}")

        # Literal data
        self.literals_num, self.attr_relations_num = self.load_literals_and_attr_relations_num()
        # For the paper we only consider numerical literals
        # self.literals_txt, self.attr_relations_txt = self.load_literals_and_attr_relations_txt()

    def load_dataframes(self):
        print('Start loading dataframes')
        # df_triples_train = pd.read_csv(osp.join(self.triple_file, 'train.txt'), header=None, sep='\t')
        # df_triples_val = pd.read_csv(osp.join(self.triple_file, 'valid.txt'), header=None, sep='\t')
        # df_triples_test = pd.read_csv(osp.join(self.triple_file, 'test.txt'), header=None, sep='\t')
        df_literals_num = pd.read_csv(osp.join(self.triple_file, 'numerical_literals_decimal.txt'), header=None, sep='\t')
        # df_literals_txt = pd.read_csv(osp.join(self.triple_file, 'text_literals.txt'), header=None, sep='\t')

        return None, None, None, df_literals_num, None # df_literals_txt

    def load_relational_data(self):
        print('Start loading relational data')
        self.entities = list(set(np.concatenate([self.df_triples_train[0].unique(),
                                                 self.df_triples_test[0].unique(),
                                                 self.df_triples_val[0].unique(),
                                                 self.df_triples_train[2].unique(),
                                                 self.df_triples_test[2].unique(),
                                                 self.df_triples_val[2].unique(),
                                                 self.df_literals_num[0].unique(),
                                                 # self.df_literals_txt[0].unique()
                                                 ])))

        self.relations = list(set(np.concatenate([self.df_triples_train[1].unique(),
                                                  self.df_triples_test[1].unique(),
                                                  self.df_triples_val[1].unique()])))

        return self.entities, self.relations

    def load_edge_indices(self):
        # Also add inverse triples to the training set
        # Concatenate edge indices of subject and object
        edge_index_train_concat = torch.cat([torch.tensor(self.df_triples_train[0].map(self.entity2id)), # subject
                                             torch.tensor(self.df_triples_train[2].map(self.entity2id))]) # object
        # And vice versa to get also inverse triples
        edge_index_train_concat_inv = torch.cat([torch.tensor(self.df_triples_train[2].map(self.entity2id)), # subject
                                                 torch.tensor(self.df_triples_train[0].map(self.entity2id))]) # object

        edge_index_train = torch.stack([edge_index_train_concat, edge_index_train_concat_inv])

        edge_index_val = torch.stack([torch.tensor(self.df_triples_val[0].map(self.entity2id)),
                                      torch.tensor(self.df_triples_val[2].map(self.entity2id))])
        edge_index_test = torch.stack([torch.tensor(self.df_triples_test[0].map(self.entity2id)),
                                       torch.tensor(self.df_triples_test[2].map(self.entity2id))])

        return edge_index_train, edge_index_val, edge_index_test

    def load_edge_types(self):
        # Concatenate relations with itself because of inverse triples (inverse relations as index + num_relations)
        edge_type_train_concat = torch.cat([torch.tensor(self.df_triples_train[1].map(self.relation2id)),
                                            torch.tensor(self.df_triples_train[1].map(self.relation2id))]) # inverse relations as index + num_relations
        edge_type_val = torch.tensor(self.df_triples_val[1].map(self.relation2id))
        edge_type_test = torch.tensor(self.df_triples_test[1].map(self.relation2id))

        return edge_type_train_concat, edge_type_val, edge_type_test

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

    # def load_literals_and_attr_relations_txt(self):
    #     print('Start loading textual literals: E x R x V')
    #     attr_relations_txt_unique = list(self.df_literals_txt[1].unique())
    #     attr_relation_txt_2_id = {attr_relations_txt_unique[i]: i for i in range(len(attr_relations_txt_unique))}
    #
    #     print("Unique textual attributive relations: ", len(attr_relations_txt_unique))
    #
    #     # Map entities to ids
    #     # Drop all literals that have entities that are not in the training set
    #     self.df_literals_txt = self.df_literals_txt[self.df_literals_txt[0].isin(self.entities)]
    #     self.df_literals_txt[0] = self.df_literals_txt[0].map(self.entity2id).astype(int)
    #     # Map attributive relations to ids
    #     self.df_literals_txt[1] = self.df_literals_txt[1].map(attr_relation_txt_2_id).astype(int)
    #     # Change literal values to string
    #     self.df_literals_num[2] = self.df_literals_num[2].astype(str)
    #
    #     nlp = spacy.load('en_core_web_md')
    #
    #     # Extract embedding vectors for one textual literal value per entity and the attributive relations
    #     features_txt = []
    #     features_txt_attr = []
    #     for i in tqdm(range(len(self.entities))):
    #         df_i = self.df_literals_txt[self.df_literals_txt[0] == i]
    #
    #         features_txt_i = torch.zeros(len(attr_relations_txt_unique), self.embedding_dim)
    #         features_txt_attr_i = torch.zeros(len(attr_relations_txt_unique))
    #         for index, row in df_i.iterrows():
    #             # Textual literal values: row[1] = attributive relation index, row[2] = literal value
    #             spacy_embedding = torch.tensor(nlp(row[2]).vector)
    #             features_txt_i[int(row[1])] = spacy_embedding
    #
    #             # One-hot encoding for attributive relations
    #             features_txt_attr_i[int(row[1])] = 1
    #
    #         features_txt.append(features_txt_i)
    #         # Normalize txt_attr one hot encoding vector
    #         features_txt_attr_i -= torch.mean(features_txt_attr_i)
    #         features_txt_attr.append(features_txt_attr_i)
    #
    #     features_txt = torch.stack(features_txt)
    #     features_txt_attr = torch.stack(features_txt_attr)
    #
    #     return features_txt.squeeze(), features_txt_attr

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

    # def cluster_literals_txt(self, n_clusters=100):
    #     # Only clusters textual literals
    #
    #     print(f"Clustering textual literals with {n_clusters} clusters...")
    #     embeddings = self.literals_txt.numpy()
    #
    #     embeddings_cluster_labeled = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(embeddings)
    #
    #     # Convert text literals to one-hot encoding of clusters
    #     self.literals_txt = torch.zeros(len(self.entities), n_clusters)
    #     for i in range(len(embeddings_cluster_labeled)):
    #         self.literals_txt[i, embeddings_cluster_labeled[i]] = 1
    #
    #     # Normalize txt_attr one hot encoding vector
    #     self.literals_txt -= torch.mean(self.literals_txt, dim=0, keepdim=True)


if __name__ == '__main__':
    dataset_name = 'LitWD48K'
    if not osp.isfile(f'data/{dataset_name}/processed.pt'):
        print('Process dataset...')
        dataset = LiteralLinkPredDataset(f'data/{dataset_name}')
        torch.save(dataset, f'data/{dataset_name}/processed.pt')

    dataset = torch.load(f'data/{dataset_name}/processed.pt')

    # Save numerical literals to numpy file
    np.save(f'data/{dataset_name}/numerical_literals_rep.npy', dataset.literals_num.numpy())
    # Save textual literals to numpy file
    # np.save(f'data/{dataset_name}/text_literals_rep.npy', dataset.literals_txt.numpy())

    # Save attributive relation type literals to numpy file
    np.save(f'data/{dataset_name}/numerical_literals_rep_attr.npy', dataset.attr_relations_num.numpy())

    filtered_ds_100 = dataset
    filtered_ds_100.filter_literals_by_attr_relation_frequency(100)
    # Save frequency filtered numerical literals to numpy file (frequency threshold = 100)
    np.save(f'data/{dataset_name}/numerical_literals_rep_filtered_100.npy',
            filtered_ds_100.literals_num.numpy())

    # clustered_ds_100 = dataset
    # clustered_ds_100.cluster_literals_txt(100)

    # # Save clustered textual literals to numpy file (n_clusters = 100)
    # np.save(f'data/{dataset_name}/text_literals_rep_clustered_100_mean.npy',
    #         clustered_ds_100.literals_txt.numpy())

