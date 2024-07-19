import tarfile as tar
import torch

from typing import Callable, List, Optional
from torch_geometric.data import Data, InMemoryDataset, download_url


class LiteralLPDataset(InMemoryDataset):

    def __init__(self, root: str, split: str = "train", dataset_name: str = "FB15k-237",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):

        self.dataset_name = dataset_name


        super().__init__(root, transform, pre_transform)

        if split not in {'train', 'val', 'test'}:
            raise ValueError(f"Invalid 'split' argument (got {split})")

        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.data, self.slices = torch.load(path)
        self.numerical_literals = torch.load(f'{self.processed_dir}/numerical_literals.pt')
        self.literal_mask = torch.load(f'{self.processed_dir}/literal_mask.pt')

    @property
    def raw_dir(self) -> str:
        return f'{self.root}/{self.dataset_name}'

    @property
    def processed_dir(self) -> str:
        return f'{self.root}/{self.dataset_name}/processed'

    @property
    def raw_file_names(self) -> List[str]:
        return ['train.txt', 'valid.txt', 'test.txt', 'numerical_literals.txt']

    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def download(self):
        raise FileNotFoundError(f'Please place the {self.dataset_name} dataset (train.txt, '
                                f'valid.txt, test.txt) folder in the dataset directory manually.')

    def process(self):
        data_list, node_dict, rel_dict = [], {}, {}
        for path in self.raw_paths:
            with open(path, 'r') as f:
                data = [x.split('\t') for x in f.read().split('\n')[:-1]]

            edge_index = torch.empty((2, len(data)), dtype=torch.long)
            edge_type = torch.empty(len(data), dtype=torch.long)
            for i, (src, rel, dst) in enumerate(data):
                if src not in node_dict:
                    node_dict[src] = len(node_dict)
                if dst not in node_dict:
                    node_dict[dst] = len(node_dict)
                if rel not in rel_dict:
                    rel_dict[rel] = len(rel_dict)

                edge_index[0, i] = node_dict[src]
                edge_index[1, i] = node_dict[dst]
                edge_type[i] = rel_dict[rel]

            data = Data(edge_index=edge_index, edge_type=edge_type)
            data_list.append(data)

        for data, path in zip(data_list, self.processed_paths):
            data.num_nodes = len(node_dict)
            torch.save(self.collate([data]), path)
        torch.save(node_dict, self.processed_dir + '/node_dict.pt')
        torch.save(rel_dict, self.processed_dir + '/rel_dict.pt')

        rel_dict_attributive = {}
        with open(self.raw_paths[-1], 'r') as f:
            for line in f:
                head, rel, value = line.strip().split('\t')
                if rel not in rel_dict_attributive:
                    rel_dict_attributive[rel] = len(rel_dict_attributive)
        torch.save(rel_dict_attributive, self.processed_dir + '/rel_dict_attributive.pt')

        numerical_literals = torch.zeros(len(node_dict), len(rel_dict_attributive))

        with open(self.raw_paths[-1], 'r') as f:
            for line in f:
                head, rel, value = line.strip().split('\t')
                numerical_literals[node_dict[head], rel_dict_attributive[rel]] = float(value)

        literal_mask = (numerical_literals != 0)
        torch.save(literal_mask, self.processed_dir + '/literal_mask.pt')

        max_lit, min_lit = torch.max(numerical_literals, dim=0).values, torch.min(numerical_literals, dim=0).values
        numerical_literals = (numerical_literals - min_lit) / (max_lit - min_lit + 1e-8)
        torch.save(numerical_literals, self.processed_dir + '/numerical_literals.pt')


if __name__ == '__main__':
    dataset = LiteralLPDataset('./data', split='train', dataset_name='FB15k-237_rand_rel-20_full')