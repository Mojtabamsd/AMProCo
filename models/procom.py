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

        ### 3) Aggregate node_logits -> leaf_logits
        # We'll produce [N, num_leaves]
        num_leaves = len(self.leaf_node_ids)
        device = node_logits.device
        leaf_logits = torch.zeros(batch_size, num_leaves, device=device)

        # Example approach: for each leaf L, sum node_logits over L's path
        root_weight = 0
        super_weight = 0.5
        leaf_weight = 1.0

        for leaf_idx, leaf_id in enumerate(self.leaf_node_ids):
            path_nodes = self.leaf_path_map[leaf_id]  # e.g. [0, 3, 17,  ... leaf_id]
            assert len(path_nodes) == 3, "Expect 3 nodes in each path for CIFAR-100"

            # Extract the node logits
            root_logit = node_logits[:, path_nodes[0]]  # shape [N]
            super_logit = node_logits[:, path_nodes[1]]
            leaf_logit = node_logits[:, path_nodes[2]]

            weighted_sum = (root_weight * root_logit +
                            super_weight * super_logit +
                            leaf_weight * leaf_logit)

            total_weight = (root_weight + super_weight + leaf_weight)

            leaf_logits[:, leaf_idx] = node_logits[:, path_nodes].sum(dim=1)

            leaf_logits[:, leaf_idx] = weighted_sum / total_weight

        return leaf_logits


    def _make_multi_hot(self, leaf_labels):
        """
        leaf_labels: shape [N] with the leaf ID for each sample
        We'll return a [N, num_nodes] multi-hot assignment
        so each sample updates the path node distributions.
        """
        N = leaf_labels.shape[0]
        device = leaf_labels.device

        multi_hot = torch.zeros(N, self.num_nodes, device=device)
        for i in range(N):
            leaf_id = leaf_labels[i].item()
            path_nodes = self.leaf_path_map[leaf_id]
            multi_hot[i, path_nodes] = 1.0

        return multi_hot







