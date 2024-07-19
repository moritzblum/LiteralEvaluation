import argparse
import os
import shutil
import random
import numpy as np
import pandas as pd

"""
Synthetic Dataset Generation

The Synthetic dataset is generated from the FB15k-237 dataset. 
The dataset is generated to test the performance of the model if both, 
structure and literal values of entities, are needed to be understood in order 
to make correct predictions. 
"""

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Synthetic', help='Name of the dataset.')
parser.add_argument('--source_dataset', type=str, default='FB15k-237', help='Name of the source dataset.')
parser.add_argument('--class_mapping_file', type=str, default='data/FB15k-237/FB15k-237_class_mapping.csv',
                    help='Path to the class mapping file that has the following columns/format: id;freebase-id;wikidata-id;label;class')
parser.add_argument('--relevant_class_label', type=str, default='human', help='Class label of the relevant entities.')

args = parser.parse_args()

# Constants
DATASET_NAME = args.dataset_name
SOURCE_DATASET = args.source_dataset
CLASS_MAPPING_FILE = args.class_mapping_file
RELEVANT_CLASS_LABEL = args.relevant_class_label

LITERAL_RELATION = '/m/has_value'
RELATIONAL_RELATION = '/m/is_a'
CLASS_HIGH = '/m/high'
CLASS_LOW = '/m/low'


if __name__ == '__main__':

    if 'dataset_name' in os.listdir(f'../data/'):
        print('Attention: the directory is not empty. This will overwrite existing files.')
    else:
        os.mkdir(f'data/{DATASET_NAME}')
        os.mkdir(f'data/{DATASET_NAME}/literals')
        print('Directory created.')

    # load training file and mapping dataframe
    train_triples = pd.read_csv(f'./data/{SOURCE_DATASET}/train.txt', sep='\t', header=None)
    class_mapping_df = pd.read_csv(CLASS_MAPPING_FILE, sep=';', header=0)

    # collect relevant URIs, i.e. all URIs that are human entities
    uris = train_triples[0].unique().tolist() + train_triples[2].unique().tolist()
    human_uris = [x for x in class_mapping_df[class_mapping_df["class_label"] == RELEVANT_CLASS_LABEL][
        "dataset_entity"].to_list() if
                  x in uris]

    num_humans = len(human_uris)
    print('Number of human entities:', num_humans)
    print('Human entities, e.g.:', human_uris[:5])

    # randomly sample literal values for all URIs
    uri_to_value = {uri: np.random.rand() for uri in uris}

    # split URIs into train, test, and validation
    # (this split just assigns where to add the /m/is_a triples)
    random.shuffle(human_uris)
    uris_split = {'test': human_uris[:int(num_humans * 0.15)],
                  'valid': human_uris[int(num_humans * 0.15):int(num_humans * 0.15) * 2],
                  'train': human_uris[int(num_humans * 0.15) * 2:]}

    # create /m/has_value triples and /m/is_a (depending on the sampled values - threshold 0.5)
    # triples for all human entities

    relational_triples = {}
    for split, uris in uris_split.items():
        relational_triples[split] = []
        for uri in uris:
            relational_triples[split].append(
                (uri, RELATIONAL_RELATION, CLASS_HIGH if uri_to_value[uri] > 0.5 else CLASS_LOW))
    print('Created relational triples, e.g.:', relational_triples['train'][:5])

    # dump relational train triples to file
    with open(f'data/{DATASET_NAME}/train_value.txt', 'w') as f:
        for triple in relational_triples['train']:
            f.write("\t".join(triple) + "\n")

    attributive_triples = []
    for uri, value in uri_to_value.items():
        attributive_triples.append((uri, LITERAL_RELATION, str(value)))
    print('Created literal triples, e.g.:', attributive_triples[:5])

    # dump literal triples to file
    with open(f'data/{DATASET_NAME}/literals/numerical_literals.txt', 'w') as f:
        for triple in attributive_triples:
            f.write("\t".join(triple) + "\n")

    # generate relational test triples
    # Attention: We will be testing the model e.g. on both human and non-human entities, which includes
    # false triples. Therefore, calculating standard metrics such as MRR(Mean Reciprocal Rank) or
    # HITS@1 not yield meaningful results.
    test_relational = []

    # neg_samples are random non-human entities that should not be assigned to /m/low or /m/high
    neg_samples = class_mapping_df[class_mapping_df["class_label"] != RELEVANT_CLASS_LABEL][
        'dataset_entity'].unique().tolist()
    random.shuffle(neg_samples)

    # generate the /m/is_a triples for the relevant entities
    # * for the human entities, we are going to investigate
    # if /m/low or /m/high is assigned correctly -> investigate literal understanding
    # * for the non-human entities, we are going to investigate if /m/low or /m/high is
    # not assigned as this never occurred in the training data -> investigate literal &
    # structural understanding
    for entity in uris_split['test'] + neg_samples[:675]:
        test_relational.append((entity, RELATIONAL_RELATION, CLASS_LOW))
        test_relational.append((entity, RELATIONAL_RELATION, CLASS_HIGH))

    # dump relational test triples to file
    with open(f'data/{DATASET_NAME}/test_value.txt', 'w') as f:
        for triple in test_relational:
            f.write("\t".join(triple) + "\n")

    # add FB15k-237 triples to the dataset
    for split in ['train', 'test', 'valid']:
        shutil.copyfile(f'./data/FB15k-237/{split}.txt', f'./data/{DATASET_NAME}/{split}.txt')

    for split in ['train', 'test']:
        with open(f'./data/{DATASET_NAME}/{split}_value.txt') as train_value_file:
            with open(f'./data/{DATASET_NAME}/{split}.txt', 'a') as train_file:
                train_file.write(train_value_file.read())

        os.remove(f'./data/{DATASET_NAME}/{split}_value.txt')
