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

Example usage: python -u main_transea.py --model transea --dataset FB15k-237_rand_rel-20_full --alpha 0.3 --lr 0.001 --hidden 100 --saved_models_path="./saved_models"
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
                       f'ranks_{split}_evaluation_TransEA_{ALPHA}_{DATASET}.tsv'),
              'w') as f:
        for head, relation, tail, head_rank, tail_rank, score in table:
            f.write("\t".join(
                [inv_node_dict[head], inv_rel_dict[relation], inv_node_dict[tail], str(head_rank),
                 str(tail_rank), str(score)]) + '\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=model_map.keys(), type=str.lower,
                        required=True)
    parser.add_argument('--data_path', choices=model_map.keys(), type=str.lower,
                        default='./data')
    parser.add_argument('--dataset', default='FB15k-237')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="ALPHA balances between KGC loss and literal prediction loss")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=50)
    parser.add_argument('--saved_models_path', type=str, default='./',
                        help='file path to the directory where trained models and predictions are saved')
    args = parser.parse_args()
    ALPHA = args.alpha
    LR = args.lr
    HIDDEN = args.hidden
    DATASET = args.dataset
    BATCH_SIZE = args.batch_size
    DATA_PATH = args.data_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = osp.join(DATA_PATH, DATASET)
    print('dataset path:', path)

    train_dataset = LiteralLPDataset(DATA_PATH, split='train', dataset_name=DATASET)
    train_data = LiteralLPDataset(DATA_PATH, split='train', dataset_name=DATASET)[0].to(device)
    val_data = LiteralLPDataset(DATA_PATH, split='val', dataset_name=DATASET)[0].to(device)
    test_data = LiteralLPDataset(DATA_PATH, split='test', dataset_name=DATASET)[0].to(device)

    node_dict = torch.load(osp.join(path, 'processed', 'node_dict.pt'))
    rel_dict = torch.load(osp.join(path, 'processed', 'rel_dict.pt'))

    inv_node_dict = {idx: uri for uri, idx in node_dict.items()}
    inv_rel_dict = {idx: uri for uri, idx in rel_dict.items()}

    literal_features = train_dataset.numerical_literals
    literal_mask = train_dataset.literal_mask
    literal_feature_dim = literal_features.size(1)

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

    print('Start training.')

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
