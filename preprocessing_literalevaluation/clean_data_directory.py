import argparse
import logging
import shutil
import os

"""
Deletes all KGA literal evaluation datasets with the given parameters.
"""
logging.getLogger().setLevel(logging.INFO)
parser = argparse.ArgumentParser(
    description='Create literals and append to graph'
)
parser.add_argument('--dataset', default='YAGO3-10', metavar='',
                    help='which dataset in {`YAGO3-10`, `FB15k`, `FB15k-237`} to be used? (default: YAGO3-10)')
parser.add_argument('--relation_ablation', default=0, type=int,
                    help='In case of literal evaluation: Percentage of relational triples to remove from the training set, e.g. 90')
parser.add_argument('--features', default='org',
                    help='In case of literal evaluation: The type of features to use. Default is `org` (original features). Other options are `rand` (random features), or `attribute` (attribute features).')
parser.add_argument('--literal_replacement_mode', default='full',
                    help='In case of literal evaluation: full or sparse.')
parser.add_argument('--run_id', default='0', type=str,
                    help='In case of literal evaluation: The run ID for the experiment to create a unique folder. Default is 0.')
args = parser.parse_args()

literalevaluation_dataset_name =  f'{args.dataset}_{args.features}_rel-{args.relation_ablation}_{args.literal_replacement_mode}_run-{args.run_id}'
logging.log(logging.INFO, f'Deleting all KGA literal evaluation datasets: {literalevaluation_dataset_name}')
for item in os.listdir('./data'):
    if literalevaluation_dataset_name in item:
        logging.log(logging.INFO, f'Deleting dataset: {item}')
        shutil.rmtree(os.path.join('./data', item))


