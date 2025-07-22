import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ive
import numpy as np
import torch.distributed as dist


class HierarchicalProCoWrapper(nn.Module):
    def __init__(self,
                 proco_loss: nn.Module,
                 leaf_node_ids: list,
                 leaf_path_map: dict,
                 num_nodes: int,
                 device='cuda'):
        """
        proco_loss: an instance of ProCoLoss (modified to have 'num_classes' = num_nodes).
        leaf_node_ids: list of the node IDs that correspond to leaves (e.g. [21..120] for CIFAR).
        leaf_path_map: dict { leaf_id -> [list_of_node_ids_in_path] }
        num_nodes: total number of nodes in the hierarchy (root + superclasses + leaves).
        """
        super().__init__()
        self.proco_loss = proco_loss  # it has EstimatorCV with dimension = num_nodes
        self.leaf_node_ids = leaf_node_ids
        self.leaf_path_map = leaf_path_map
        self.num_nodes = num_nodes
        self.device = device

        num_leaves = len(leaf_node_ids)
        leaf2node = torch.zeros(num_leaves, num_nodes, dtype=torch.bool)
        proto_lists = []
        root_ids = []
        leaf_ids = []

        for li, leaf in enumerate(leaf_node_ids):
            path = leaf_path_map[leaf]
            leaf2node[li, path] = True
            root_ids.append(path[0])
            leaf_ids.append(path[-1])
            proto_lists.append(torch.tensor(path[1:-1], dtype=torch.long))

        self.register_buffer("leaf2node_mask", leaf2node)
        self.register_buffer("root_ids", torch.tensor(root_ids))
        self.register_buffer("leaf_ids", torch.tensor(leaf_ids))

        self.proto_lists = proto_lists

    def _make_multi_hot(self, leaf_labels: torch.Tensor) -> torch.Tensor:
        """
        leaf_labels : (N,)  int64   – leaf ID indices in [0 … num_leaves)
        returns     : (N, num_nodes)  float32 multi-hot rows
        """
        return self.leaf2node_mask[leaf_labels].float()

    def forward(self, features: torch.Tensor, leaf_labels=None):
        """
        features: (N, D) already ℓ2-normalised
        leaf_labels: (N,)   leaf indices *matching the order* in leaf_node_ids
        """

        if leaf_labels is not None:
            multi_hot = self._make_multi_hot(leaf_labels)
            self.proco_loss.estimator_old.update_CV(features.detach(), multi_hot)
            self.proco_loss.estimator.update_CV(features.detach(), multi_hot)
            self.proco_loss.estimator_old.update_kappa()

        node_logits = self.proco_loss(features, labels=None)

        N, num_leaves = features.size(0), len(self.leaf_node_ids)
        device = features.device
        leaf_logits = torch.empty(N, num_leaves, device=device)

        root_log = node_logits[:, self.root_ids]
        leaf_log = node_logits[:, self.leaf_ids]

        for li, proto_idx in enumerate(self.proto_lists):
            if proto_idx.numel() == 0:                        # degenerate
                best_proto = 0.0
            elif proto_idx.numel() == 1:                      # 1 proto
                best_proto = node_logits[:, proto_idx[0]]
            else:                                             # ≥2
                best_proto, _ = torch.max(node_logits[:, proto_idx], dim=1)
            leaf_logits[:, li] = root_log[:, li] + best_proto + leaf_log[:, li]

        return leaf_logits






