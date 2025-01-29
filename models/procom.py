import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ive
import numpy as np
import torch.distributed as dist


class HierarchicalProCoWrapper1(nn.Module):
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
        super_weight = 1
        leaf_weight = 1

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

        # 3) Hard Negative Mining => Build multi-label target
        if leaf_labels is not None:
            targets = build_hard_neg_targets(node_logits, leaf_labels, self.leaf_path_map, self.K)
            # compute a multi-label BCE or margin loss
            hard_neg_loss_val = hard_neg_loss(node_logits, targets)
        else:
            hard_neg_loss_val = None  # or 0.0

        return hard_neg_loss_val

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


def build_hard_neg_targets(node_logits, leaf_label, leaf_path_map, K):
    """
    node_logits: [batch, num_nodes]
    leaf_label:  [batch] - each is in [0..99]
    Returns multi-label target => shape [batch, num_nodes], where
       +1 for path nodes,
       0  for everything else except top-K negative nodes -> we can assign 0 for them too,
       or we treat them with a strong negative label if you prefer.
    """
    device = node_logits.device
    batch_size, num_nodes = node_logits.shape
    targets = torch.zeros(batch_size, num_nodes, device=device)

    for i in range(batch_size):
        leaf_id = leaf_label[i].item()
        pos_nodes = set(leaf_path_map[leaf_id])
        # set them to +1
        for n in pos_nodes:
            targets[i, n] = 1.0

        # find negative nodes
        neg_nodes = list(set(range(num_nodes)) - pos_nodes)

        # pick top-K from node_logits[i, neg_nodes]
        neg_vals = node_logits[i, neg_nodes]  # shape [#neg_nodes]
        # get top K indices in neg_vals
        # We want the K with largest logit => "most likely negative"
        topk_vals, topk_idx = torch.topk(neg_vals, k=K)
        for j in topk_idx:
            node_id = neg_nodes[j.item()]
            # set them to 0 if you want standard multi-label
            # or set them to -1 if you'd like a different representation
            targets[i, node_id] = 0.0  # or some negative indicator

    return targets


def hard_neg_loss(node_logits, targets):
    """
    node_logits: [batch, num_nodes]
    targets: [batch, num_nodes], each entry in {0,1} or {0,1,-1} depending on your scheme
    We'll do a simple BCE:
       if target=1 => -log(sigmoid(node_logit))
       if target=0 => -log(1 - sigmoid(node_logit))
    Then sum/mean over the "labeled" positions.
    """
    # if you used -1 for "strong negative," you might do a custom transform, e.g.:
    #   t' = max(0, target), or something. We'll keep it simple (0 or 1).
    # Ensure targets are in {0,1}
    # for i in range(len(targets)): # if -1, set to 0, etc.

    bce = F.binary_cross_entropy_with_logits(node_logits, targets, reduction='none')
    # you can do a mask if you only want to average over the positions that are 1 or top-K
    # e.g. mask = (targets != 0) # if 0 means unlabeled
    # bce = bce * mask
    return bce.mean()  # or sum()

