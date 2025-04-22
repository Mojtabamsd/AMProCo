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
                 temperature: float = 1.0,
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
        self.temperature = temperature
        self.device = device

    def forward(self, features, leaf_labels=None):
        """
        1) If leaf_labels is not None, we do memory updates in the 'EstimatorCV'
           for all node-level distributions (multi-hot).
        2) Then we get the node-level logits from ProCoLoss ( shape [N, num_nodes] ).
        3) Convert node-level logits -> leaf-level logits by summing path-based scores.
        4) Return leaf-level logits => can be passed to logit adjustment or cross-entropy.
        """
        batch_size = features.shape[0]

        ### 1) Build multi-hot or path-based assignment for the memory update
        ###    so that each sample updates the path node distributions.
        if leaf_labels is not None:
            # We'll gather the path for each sample => multi-hot for all nodes
            multi_hot = self._make_multi_hot(leaf_labels)
            # The original ProCoLoss expects "labels" as shape [N], but we adapt:
            # We'll pass None to forward() but do the update directly in the Estimator...
            self.proco_loss.estimator_old.update_CV(features.detach(), multi_hot)
            self.proco_loss.estimator.update_CV(features.detach(), multi_hot)
            self.proco_loss.estimator_old.update_kappa()

        ### 2) Evaluate the node-level "contrast_logits" the same way your code does.
        #    We call the ProCoLoss forward with labels=None so it doesn't do the standard single-label scatter.
        node_logits = self.proco_loss(features, labels=None)
        # shape: [N, num_nodes], each entry is the log-likelihood ratio or partial.

        # Combine prototypes at inference
        N = features.size(0)
        num_leaves = len(self.leaf_node_ids)
        leaf_logits = torch.zeros(N, num_leaves, device=features.device)

        for leaf_idx, leaf_id in enumerate(self.leaf_node_ids):

            path_nodes = self.leaf_path_map[leaf_id]  # e.g. [root, p0, p1, leaf] (2 prototypes => 4 nodes)
            root_log = node_logits[:, path_nodes[0]]
            proto_logs = [node_logits[:, pid] for pid in path_nodes[1:-1]]
            leaf_log = node_logits[:, path_nodes[-1]]

            # Best match or mixture?
            best_proto_log, _ = torch.max(torch.stack(proto_logs, dim=1), dim=1)  # shape [N]
            leaf_logits[:, leaf_idx] = root_log + best_proto_log + leaf_log

        return leaf_logits

    def _make_multi_hot(self, leaf_labels):
        """
        For each sample i:
          - retrieve leaf_id
          - get path = leaf_path_map[leaf_id] => e.g. [root, protoX, protoY, leaf_id]
          - set multi_hot[i, path] = 1
        """
        N = leaf_labels.size(0)
        multi_hot = torch.zeros(N, self.num_nodes, device=leaf_labels.device)
        for i in range(N):
            leaf_id = leaf_labels[i].item()
            path_nodes = self.leaf_path_map[leaf_id]
            multi_hot[i, path_nodes] = 1
        return multi_hot




