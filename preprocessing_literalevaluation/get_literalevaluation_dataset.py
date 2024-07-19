import collections
import logging
import os
import shutil
import random
import numpy as np
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_ablated_dataset(dataset,
                           features='org',
                           literal_replacement_mode='sparse',
                           relation_ablation=0,
                           data_path='../data',
                           run_id="0",
                           **kwargs):
    """
    This function creates an ablated dataset for literal evaluation based on an existing Link Prediction
    dataset with literals.

    :param dataset: Source dataset, e.g. 'FB15k-237', 'LitWD48K', 'YAGO3-10', 'Synthetic', stored in the
    input_data_dir.
    :param features: The type of features to use. Default is 'org' (original features). Other options
    are 'rand' (random features), or 'attribute' (attributive relation type only).
    :param literal_replacement_mode: The mode for replacing literals. Default is 'sparse'. In 'sparse'
    mode, only existing literal values are replaced with random values. In 'full' mode, all entities get
    all attributive relations with one random value.
    :param relation_ablation: The amount of relational triples to remove from the dataset.
    Default is 0 (no removal).
    :param data_path: The directory where the dataset is stored.
    :param run_id: The run ID for the experiment to create a unique folder. Default is 0.
    :param kwargs: Additional arguments for the random features file creation.
    E.g. max_num_attributive_relations that can be used in the 'full' literal replacement mode
    to limit the number of attributive relation types to reduce the numer of attributive triples.

    :return: New ablated dataset named: {dataset}_{features}_rel-{relation_ablation}{run_id}
    """
    output_dataset = f'{dataset}_{features}_rel-{relation_ablation}_{literal_replacement_mode}{run_id}'
    output_path = os.path.join(data_path, output_dataset)

    if os.path.exists(output_path):
        logging.log(logging.INFO, f'Dataset {output_path} already exists. Skipping creation.')
        return output_dataset
    else:
        logging.log(logging.INFO, f'Creating dataset {output_path}.')

    os.makedirs(output_path)

    # Copy the train file. In case of relation ablation, the train file is pruned.
    if relation_ablation > 0:
        create_relation_ablation_train_file(os.path.join(data_path, dataset, 'train.txt'),
                                            output_path,
                                            relation_ablation_percent=relation_ablation)
    else:
        shutil.copyfile(os.path.join(data_path, dataset, 'train.txt'),
                        os.path.join(output_path, 'train.txt'))

    # Copy valid and test files in any case.
    shutil.copyfile(os.path.join(data_path, dataset, 'valid.txt'),
                    os.path.join(output_path, 'valid.txt'))
    shutil.copyfile(os.path.join(data_path, dataset, 'test.txt'), os.path.join(output_path, 'test.txt'))

    # Copy the literal file or create a new literal files with random values.
    if features == 'org':
        shutil.copyfile(os.path.join(data_path, dataset, 'numerical_literals.txt'),
                        os.path.join(output_path, 'numerical_literals.txt'))
    elif features == 'rand':
        create_random_features_file(os.path.join(data_path, dataset, 'numerical_literals.txt'),
                                    output_path,
                                    mode=literal_replacement_mode,
                                    **kwargs)
    elif features == 'attribute':
        create_attribute_features_file(output_path,
                                       os.path.join(data_path, dataset, 'numerical_literals.txt'))
    else:
        raise Exception(f'Features {features} not supported.')

    return output_dataset


def create_attribute_features_file(data_path, source_literal_file):
    """
    This function modifies an existing literal file by setting all literal values to 1.0.
    :param data_path: The directory where the dataset is stored.
    :param source_literal_file: The original literal file.
    :return:
    """
    with open(source_literal_file) as f:
        with open(os.path.join(data_path, 'numerical_literals.txt'), 'w') as out_file:
            for line in f:
                head, rel, tail = line.strip().split('\t')
                out_file.write('\t'.join([head, rel, '1.0']) + '\n')


def create_random_features_file(source_literal_file, data_path,
                                mode='full', **kwargs):
    """
    This function creates a literal file with random features.

    :param source_literal_file: The original literal file.
    :param data_path: The directory where the dataset is stored.
    :param mode: full or sparse - full means all entities get all attributes, sparse means only the
    existing attributes are replaced with random values s.t. the number of attributes is the same.
    :param max_num_attributive_relations: Can be used in the 'full' literal replacement mode
    to limit the number of attributive relation types to reduce the numer of attributive triples.

    :return:
    """
    if mode == 'full':
        entities = set()
        for split in ['train', 'valid', 'test']:
            with open(os.path.join(data_path, f'{split}.txt')) as f:
                for line in f:
                    head, rel, tail = line.strip().split('\t')
                    entities.add(head)
                    entities.add(tail)

        attributive_relations = set()
        with open(source_literal_file) as f:
            for line in f:
                head, rel, tail = line.strip().split('\t')
                attributive_relations.add(rel)

        logging.log(logging.INFO,
                    f'Num entities: {len(entities)}, Num attributive relations: {len(attributive_relations)}')

        if "max_num_attributive_relations" in kwargs:
            if kwargs["max_num_attributive_relations"]:
                print(kwargs["max_num_attributive_relations"])
                attributive_relations = list(attributive_relations)[:kwargs["max_num_attributive_relations"]]
                logging.log(logging.INFO,
                            f'Too many attributive relations. Reducing to {len(attributive_relations)}.')

        with open(os.path.join(data_path, 'numerical_literals.txt'), 'w') as out_file:
            for entity in entities:
                for rel in attributive_relations:
                    out_file.write('\t'.join([entity, rel, str(random.uniform(0, 1))]) + '\n')

    elif mode == 'sparse':
        with open(source_literal_file) as f:
            with open(os.path.join(data_path, 'numerical_literals.txt'), 'w') as out_file:
                for line in f:
                    head, rel, tail = line.strip().split('\t')
                    out_file.write('\t'.join([head, rel, str(random.uniform(0, 1))]) + '\n')
    else:
        raise Exception(f'Mode {mode} not supported.')


def create_relation_ablation_train_file(source_train_file, data_path,
                                        relation_ablation_percent=10):
    """
    This function creates a pruned training file by removing a certain percentage of relational triples.

    :param source_train_file: The original train.txt file containing the relational triples.
    :param data_path: The directory where the dataset is stored.
    :param relation_ablation_percent: The percentage of triples to remove from the source training set.
    """
    uri_2_freq = []
    triples = []
    with open(source_train_file) as train_in:
        for line in train_in:
            triple = line.rstrip().split('\t')
            uri_2_freq.extend(triple)
            triples.append(triple)
    uri_2_freq = collections.Counter(uri_2_freq)
    num_triples = len(triples)

    del_num = int((relation_ablation_percent / 100) * num_triples)

    permuted_triples = [triples[i] for i in np.random.permutation(np.arange(0, num_triples))]

    triples_maintained = []
    for head, relation, tail in permuted_triples:
        if del_num >= 1 and uri_2_freq[head] > 1 and uri_2_freq[relation] > 1 and uri_2_freq[tail] > 1:
            del_num -= 1
            continue
        else:
            triples_maintained.append([head, relation, tail])

    with open(os.path.join(data_path, 'train.txt'), 'w') as pruned_out:
        for triple in triples_maintained:
            pruned_out.write('\t'.join(triple) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='YAGO3-10')
    parser.add_argument('--features', default='org')
    parser.add_argument('--literal_replacement_mode', default='sparse')
    parser.add_argument('--relation_ablation', default=0, type=int)
    parser.add_argument('--max_num_attributive_relations', type=int)
    parser.add_argument('--data_dir', default='../data', type=str)
    parser.add_argument('--run_id', default='', type=str,
                        help="Run ID for the experiment. Can be used if multiple versions of the same dataset must be created.")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    RUN_ID = '_run-' + args.run_id if args.run_id != '' else ''

    create_ablated_dataset(
        args.dataset,
        features=args.features,
        relation_ablation=args.relation_ablation,
        literal_replacement_mode=args.literal_replacement_mode,
        max_num_attributive_relations=args.max_num_attributive_relations,
        run_id=RUN_ID,
        data_path=DATA_DIR)
