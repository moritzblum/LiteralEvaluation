import tarfile as tar
import torch

from typing import Callable, List, Optional
from torch_geometric.data import Data, InMemoryDataset, download_url


class LiteralLPDataset(InMemoryDataset):

    url = {
        "FB15k-237": 'https://github.com/SmartDataAnalytics/LiteralE/raw/master/datasets/FB15k-237.tar.gz',
        "YAGO3-10": 'https://github.com/SmartDataAnalytics/LiteralE/raw/master/datasets/YAGO3-10.tar.gz',
        "LitWD48K": 'https://github.com/GenetAsefa/LiterallyWikidata/raw/main/Datasets/LitWD48K',
        "Synthetic": 'https://github.com/moritzblum/LiteralEvaluation/raw/variations/data/Synthetic'}

    def __init__(self, root: str, split: str = "train", dataset_name: str = "FB15k-237",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):

        self.dataset_name = dataset_name
        super().__init__(root, transform, pre_transform)

        if split not in {'train', 'val', 'test'}:
            raise ValueError(f"Invalid 'split' argument (got {split})")

        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.data, self.slices = torch.load(path)

        print(self.raw_dir)

    @property
    def raw_file_names(self) -> List[str]:
        return ['train.txt', 'valid.txt', 'test.txt']

    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def download(self):

        if self.url[self.dataset_name].endswith('.tar.gz'):
            download_url(f'{self.url[self.dataset_name]}', self.raw_dir)
            file = tar.open(f'{self.raw_dir}/{self.dataset_name}.tar.gz')
            file.extractall(self.raw_dir)
            file.close()
            print('extracted')
        else:
            for raw_file_name in self.raw_file_names:
                download_url(f'{self.url[self.dataset_name]}/{raw_file_name}', self.raw_dir)

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
