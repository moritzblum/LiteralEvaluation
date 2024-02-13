import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
from pathlib import Path


parser = argparse.ArgumentParser(
    description='Create literals'
)
parser.add_argument('--dataset', default='YAGO3-10', metavar='',
                    help='which dataset in {`YAGO3-10`, `FB15k`, `FB15k-237`, LitWD48K} to be used? (default: YAGO3-10)')
args = parser.parse_args()


# Load vocab
vocab = np.load(f'{str(Path.home())}/.data/{args.dataset}/vocab_e1', allow_pickle=True)

ent2idx = vocab[0]
idx2ent = vocab[1]

numerical_literals_files = ['numerical_literals_rich.txt'] if args.dataset != 'LitWD48K' else ['numerical_literals_decimal.txt', 'numerical_literals_double.txt']

for numerical_literals_file in numerical_literals_files:
    # Load raw literals
    df = pd.read_csv(f'data/{args.dataset}/literals/{numerical_literals_file}', header=None, sep='\t')

    rel2idx = {v: k for k, v in enumerate(df[1].unique())}

    # Resulting file
    num_lit = np.zeros([len(ent2idx), len(rel2idx)], dtype=np.float32)

    # Create literal wrt vocab
    for i, (s, p, lit) in tqdm(enumerate(df.values)):
        try:
            num_lit[ent2idx[s.lower()], rel2idx[p]] = lit
        except KeyError:
            continue

    np.save(f'data/{args.dataset}/literals/{numerical_literals_file.replace(".txt", ".npy")}', num_lit)
