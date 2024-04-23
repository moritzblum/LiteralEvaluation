import argparse
import os.path as osp
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import download_url
from pyg_dataset import LiteralLPDataset
from pyg_transe import TransE, TransEA


"""
TransEA with PyG

Example usage: python -u examples/main_transea.py --model transea --dataset Synthetic --alpha 0.3 --lr 0.001 --hidden 100 --features rand --saved_models_path="/homes/mblum/LiteralEvaluation/saved_models" > ~/LiteralEvaluation/results/slurm_transea_synthetic_rand.txt
"""


model_map = {
    'transe': TransE,
    'transea': TransEA,  # the only one working with literals
}


def train():
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index, literal_features, literal_mask)

        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples


@torch.no_grad()
def test(data):
    model.eval()
    return model.test(
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=20000,
        filtered=True,
        all_graph_data=[train_data, val_data, test_data]
    )


def write_results_table(table, split):
    with open(osp.join(args.saved_models_path,
                       f'ranks_{split}_evaluation_TransEA_{ALPHA}_{DATASET}_{FEATURES.replace(".npy", "")}_train.tsv'),
              'w') as f:
        for head, relation, tail, head_rank, tail_rank, score in table:
            f.write("\t".join(
                [inv_node_dict[head], inv_rel_dict[relation], inv_node_dict[tail], str(head_rank),
                 str(tail_rank), str(score)]) + '\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=model_map.keys(), type=str.lower,
                        required=True)
    parser.add_argument('--dataset', default='FB15k-237')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="ALPHA balances between KGC loss and literal prediction loss")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=50)
    parser.add_argument('--features', type=str, default='numerical_literals.npy',
                        help='rand or file name, e.g. numerical_literals.npy')

    parser.add_argument('--saved_models_path', type=str, default='./',
                        help='file path to the directory where trained models and predictions are saved')
    args = parser.parse_args()
    ALPHA = args.alpha
    LR = args.lr
    HIDDEN = args.hidden
    FEATURES = args.features
    DATASET = args.dataset
    BATCH_SIZE = args.batch_size

    VOCAB_LINKS = {
        'LitWD48K': 'https://github.com/moritzblum/LiteralEvaluation/raw/variations/data/LitWD48K/vocab_e1',
        'FB15k-237': 'https://github.com/moritzblum/LiteralEvaluation/raw/variations/data/FB15k-237/vocab_e1',
        'YAGO3-10': 'https://github.com/moritzblum/LiteralEvaluation/raw/variations/data/YAGO3-10/vocab_e1',
        'Synthetic': 'https://github.com/moritzblum/LiteralEvaluation/raw/variations/data/Synthetic/vocab_e1'
    }

    FEATURE_LINKS = {
        'LitWD48K': 'https://github.com/moritzblum/LiteralEvaluation/raw/variations/data/LitWD48K/literals/numerical_literals_decimal.npy',
        'FB15k-237': 'https://github.com/moritzblum/LiteralEvaluation/raw/variations/data/FB15k-237/literals/numerical_literals.npy',
        'YAGO3-10': 'https://github.com/moritzblum/LiteralEvaluation/raw/variations/data/YAGO3-10/literals/numerical_literals.npy',
        'Synthetic': 'https://github.com/moritzblum/LiteralEvaluation/raw/variations/data/Synthetic/literals/numerical_literals.npy'
    }

    LITERAL_FEATURE_DIMs = {'FB15k-237': 121,
                    'YAGO3-10': 5,
                    'LitWD48K': 246,
                    'Synthetic': 1}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = osp.join('data', DATASET)
    print('dataset path:', path)

    train_data = LiteralLPDataset(path, split='train', dataset_name=DATASET)[0].to(device)
    val_data = LiteralLPDataset(path, split='val', dataset_name=DATASET)[0].to(device)
    test_data = LiteralLPDataset(path, split='test', dataset_name=DATASET)[0].to(device)

    download_url(VOCAB_LINKS[DATASET], osp.join(path, 'processed'))
    download_url(FEATURE_LINKS[DATASET], osp.join(path, 'processed'))

    # load mapping files of LiteralE and PyG
    # id -> uri
    literale_vocab = np.load(osp.join(path, 'processed', 'vocab_e1'), allow_pickle=True)[0]
    # uri -> id
    node_dict = torch.load(osp.join(path, 'processed', 'node_dict.pt'))
    rel_dict = torch.load(osp.join(path, 'processed', 'rel_dict.pt'))

    inv_node_dict = {idx: uri for uri, idx in node_dict.items()}
    inv_rel_dict = {idx: uri for uri, idx in rel_dict.items()}

    # lower case all URIs as the LiteralE URIs in vocab_e1 were lower cased
    node_dict = {uri.lower(): idx for uri, idx in node_dict.items()}
    rel_dict = {uri.lower(): idx for uri, idx in rel_dict.items()}

    literal_feature_dim = LITERAL_FEATURE_DIMs[DATASET]

    # load LiteralE literal features
    if FEATURES not in ['rand', 'zero']:
        literale_features = np.load(osp.join(path, 'processed', FEATURES))
        literal_features = torch.rand(max(node_dict.values()) + 1, literal_feature_dim)

        for uri, id in literale_vocab.items():
            if uri in node_dict:
                literal_features[node_dict[uri]] = torch.tensor(literale_features[id])
            else:
                print('Features not used by PyG:', uri)

        literal_mask = (literal_features != 0)

        # Normalize numerical literals
        max_lit, min_lit = torch.max(literal_features, dim=0).values, torch.min(literal_features,
                                                                                dim=0).values
        literal_features = (literal_features - min_lit) / (max_lit - min_lit + 1e-8)

    elif FEATURES == 'zero':
        literal_features = torch.zeros(max(node_dict.values()) + 1, literal_feature_dim)
        literal_mask = torch.ones_like(literal_features).bool()
    elif FEATURES == 'rand':
        literal_features = torch.rand(max(node_dict.values()) + 1, literal_feature_dim)
        literal_mask = torch.ones_like(literal_features).bool()
    else:
        raise ValueError('Features must be rand, zero or file name')

    literal_mask = literal_mask.to(device)
    literal_features = literal_features.to(device)

    model_arg_map = {'rotate': {'margin': 9.0}}
    model = model_map[args.model](
        num_nodes=train_data.num_nodes,
        num_relations=train_data.num_edge_types,
        hidden_channels=HIDDEN,
        lit_feature_dim=literal_feature_dim,
        alpha=ALPHA,
        **model_arg_map.get(args.model, {}),
    ).to(device)

    loader = model.loader(
        head_index=train_data.edge_index[0],
        rel_type=train_data.edge_type,
        tail_index=train_data.edge_index[1],
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    optimizer_map = {
        'transe': optim.Adam(model.parameters(), lr=LR),
        'transea': optim.Adam(model.parameters(), lr=LR),
        'complex': optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6),
        'distmult': optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6),
        'rotate': optim.Adam(model.parameters(), lr=1e-3),
    }
    optimizer = optimizer_map[args.model]

    for epoch in range(1, 501):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        if epoch % 25 == 0:
            scores, result_table = test(val_data)
            print(
                f'Eval MR: {scores["mr"]:.2f}, MRR: {scores["mrr"]:.4f}, Hits@10: {scores["hits@10"]:.4f}, Hits@3: {scores["hits@3"]:.4f}, Hits@1: {scores["hits@1"]:.4f}')

            write_results_table(result_table, 'dev')

    scores, result_table = test(test_data)

    print(
        f'Test MR: {scores["mr"]:.2f}, MRR: {scores["mrr"]:.4f}, Hits@10: {scores["hits@10"]:.4f}, Hits@3: {scores["hits@3"]:.4f}, Hits@1: {scores["hits@1"]:.4f}')

    write_results_table(result_table, 'test')
