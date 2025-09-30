import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Use built-in PyTorch cross-entropy loss
        loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')

        # Apply reduction if specified
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute the cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Calculate focal weights
        one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        focal_weights = torch.where(one_hot == 1, self.alpha * (1 - inputs.softmax(dim=1)) ** self.gamma,
                                    (1 - self.alpha) * (inputs.softmax(dim=1)) ** self.gamma)

        # Apply the focal weights to the cross-entropy loss
        focal_loss = ce_loss * focal_weights[:, 1]

        # Apply reduction if specified
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss


class LogitAdjustmentLoss(nn.Module):
    def __init__(self, weight=None):
        super(LogitAdjustmentLoss, self).__init__()
        self.class_weights = nn.Parameter(weight)

    def forward(self, logits, targets):
        # Apply logit adjustment
        adjusted_logits = logits * self.class_weights

        # Compute cross-entropy loss
        loss = F.cross_entropy(adjusted_logits, targets)

        return loss


class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None, device='cuda:0'):
        super(LogitAdjust, self).__init__()
        self.device = device
        cls_num_list = torch.FloatTensor(cls_num_list).to(self.device)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        if weight is not None:
            self.weight = weight.to(self.device)
        else:
            self.weight = None

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)


class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.5):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, preds, targets):
        errors = targets - preds
        quantile_loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        return torch.mean(quantile_loss)


class WeightedMSELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets, label):
        mse_loss = nn.MSELoss()(inputs, targets)
        weighted_mse_loss = mse_loss * self.weight[label]

        # Apply reduction if specified
        if self.reduction == 'mean':
            weighted_mse_loss = torch.mean(weighted_mse_loss)
        elif self.reduction == 'sum':
            weighted_mse_loss = torch.sum(weighted_mse_loss)

        return weighted_mse_loss


class PerceptualReconstructionLoss(nn.Module):
    def __init__(self, config, device, alpha=0.2, beta=0.8):
        super(PerceptualReconstructionLoss, self).__init__()
        self.resnet = ResNetCustom(num_classes=config.sampling.num_class,
                                   latent_dim=config.autoencoder.latent_dim,
                                   gray=config.autoencoder.gray,
                                   pretrained=True)
        self.resnet.to(device)
        self.criterion = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, generated, ground_truth):
        # Extract features from ResNet18
        _, gen_features = self.resnet(generated)
        _, gt_features = self.resnet(ground_truth)

        perceptual_loss = self.criterion(gen_features, gt_features)
        reconstruction_loss = self.criterion(generated, ground_truth)

        return self.alpha * perceptual_loss + self.beta * reconstruction_loss


class BalSCL(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1, device='cuda:0'):
        super(BalSCL, self).__init__()
        self.device = device
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def _compute_batch_centers(self, features, targets):
        """Compute class centers from the current mini-batch."""
        num_classes = len(self.cls_num_list)
        feat_dim = features.size(1)
        centers = []
        for c in range(num_classes):
            mask = (targets == c)
            if mask.any():
                centers.append(features[mask].mean(dim=0))
            else:
                centers.append(torch.zeros(feat_dim, device=features.device))
        return torch.stack(centers, dim=0)

    def forward(self, features, targets):
        device = self.device
        batch_size = features.shape[0]

        if targets.shape[0] * 2 == batch_size:
            targets = torch.cat([targets, targets], dim=0)

        # compute batch-specific centers
        centers1 = self._compute_batch_centers(features, targets)

        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(len(self.cls_num_list), device=device).view(-1, 1)
        targets = torch.cat([targets.repeat(2, 1), targets_centers], dim=0)

        batch_cls_count = torch.eye(len(self.cls_num_list), device=device)[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # class-complement
        # features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = torch.cat([features, centers1], dim=0)
        logits = features[:2 * batch_size].mm(features.T)
        logits = torch.div(logits, self.temperature)

        # numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets],
                                      device=device).view(1, -1).expand(2 * batch_size, 2 * batch_size + len(self.cls_num_list)) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)

        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num_list))
        m_list = m_list * (max_m / torch.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.s = s

    def forward(self, logit, target):
        index = torch.zeros_like(logit, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        logit_m = logit - batch_m * self.s  # scale only the margin, as the logit is already scaled.

        output = torch.where(index, logit_m, logit)
        return F.cross_entropy(output, target)