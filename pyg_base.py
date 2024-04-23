from typing import Tuple, Dict, List

import torch
from torch import Tensor
from torch.nn import Embedding
from tqdm import tqdm

from torch_geometric.nn.kge.loader import KGTripletLoader


class KGEModel(torch.nn.Module):
    r"""An abstract base class for implementing custom KGE models.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_channels = hidden_channels

        self.node_emb = Embedding(num_nodes, hidden_channels, sparse=sparse)
        self.rel_emb = Embedding(num_relations, hidden_channels, sparse=sparse)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.node_emb.reset_parameters()
        self.rel_emb.reset_parameters()

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        r"""Returns the score for the given triplet.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        raise NotImplementedError

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:
        r"""Returns the loss value for the given triplet.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        raise NotImplementedError

    def loader(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        **kwargs,
    ) -> Tensor:
        r"""Returns a mini-batch loader that samples a subset of triplets.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
            **kwargs (optional): Additional arguments of
                :class:`torch.utils.data.DataLoader`, such as
                :obj:`batch_size`, :obj:`shuffle`, :obj:`drop_last`
                or :obj:`num_workers`.
            """
        return KGTripletLoader(head_index, rel_type, tail_index, **kwargs)

    @torch.no_grad()
    def test(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        batch_size: int,
        log: bool = True,
        filtered: bool = False,
        all_graph_data: list = None,
    ) -> Tuple[Dict[str, float], List]:
        r"""Evaluates the model quality by computing Mean Rank and
        Hits @ :math:`k` across all possible tail entities.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
            batch_size (int): The batch size to use for evaluating.
            k (int, optional): The :math:`k` in Hits @ :math:`k`.
                (default: :obj:`10`)
            log (bool, optional): If set to :obj:`False`, will not print a
                progress bar to the console. (default: :obj:`True`)
        """
        arange = range(head_index.numel())
        arange = tqdm(arange) if log else arange

        head_ranks = []
        tail_ranks = []
        test_triple_scores = []
        for i in arange:
            h, r, t = head_index[i], rel_type[i], tail_index[i]

            # Try all nodes as tails, but delete true triplets:
            scores = []
            tail_indices = torch.arange(self.num_nodes, device=t.device)

            if filtered:
                tail_mask = torch.ones_like(tail_indices, dtype=torch.bool)
                for (heads, tails), types in [
                    (d.edge_index, d.edge_type) for d in all_graph_data
                ]:
                    tail_mask[tails[(heads == h) & (types == r)]] = False

                tail_mask[t] = True
            else:
                tail_mask = torch.ones_like(tail_indices, dtype=torch.bool)

            for ts in tail_indices[tail_mask].split(batch_size):
                scores.append(self(h.expand_as(ts), r.expand_as(ts), ts))

            score_tail_variation = torch.cat(scores)[tail_mask[:t].sum()]
            test_triple_scores.append(score_tail_variation.cpu().item())
            rank = int((torch.cat(scores).argsort(descending=True) == tail_mask[:t].sum()).nonzero().view(
                -1))
            tail_ranks.append(rank + 1)

            # Try all nodes as heads, but delete true triplets:
            scores = []
            head_indices = torch.arange(self.num_nodes, device=h.device)

            if filtered:
                head_mask = torch.ones_like(head_indices, dtype=torch.bool)
                for (heads, tails), types in [
                    (d.edge_index, d.edge_type) for d in all_graph_data
                ]:
                    head_mask[heads[(tails == t) & (types == r)]] = False

                head_mask[h] = True
            else:
                head_mask = torch.ones_like(head_indices, dtype=torch.bool)

            for hs in head_indices[head_mask].split(batch_size):
                scores.append(self(hs, r.expand_as(hs), t.expand_as(hs)))

            rank = int((torch.cat(scores).argsort(descending=True) == head_mask[:h].sum()).nonzero().view(
                -1))
            head_ranks.append(rank + 1)



        result_table = []
        for i in range(len(head_index)):
            result_table.append([head_index[i].item(), rel_type[i].item(), tail_index[i].item(), head_ranks[i], tail_ranks[i], test_triple_scores[i]])

        ranks = torch.tensor(tail_ranks + head_ranks, dtype=torch.float)
        metrics = {f'hits@{k}': ranks.le(k).float().mean() for k in [1, 3, 5, 10]}
        metrics['mr'] = ranks.mean()
        metrics['mrr'] = ranks.reciprocal().mean()

        return metrics, result_table

    @torch.no_grad()
    def random_sample(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Randomly samples negative triplets by either replacing the head or
        the tail (but not both).

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        # Random sample either `head_index` or `tail_index` (but not both):
        num_negatives = head_index.numel() // 2
        rnd_index = torch.randint(self.num_nodes, head_index.size(),
                                  device=head_index.device)

        head_index = head_index.clone()
        head_index[:num_negatives] = rnd_index[:num_negatives]
        tail_index = tail_index.clone()
        tail_index[num_negatives:] = rnd_index[num_negatives:]

        return head_index, rel_type, tail_index

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'num_relations={self.num_relations}, '
                f'hidden_channels={self.hidden_channels})')
