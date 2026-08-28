import math
import torch
import torch.nn as nn


class NCC:
    def __init__(self, eps=1e-5):
        self.eps = eps

    def __call__(self, x, y, weight=None):

        if weight is None:
            weight = torch.ones_like(x)

        mu_x = (weight * x).sum() / weight.sum()
        mu_y = (weight * y).sum() / weight.sum()

        x_centered = x - mu_x
        y_centered = y - mu_y

        num = (weight * x_centered * y_centered).sum()
        denom1 = torch.sqrt((weight * torch.square(x_centered)).sum() + self.eps)
        denom2 = torch.sqrt((weight * torch.square(y_centered)).sum() + self.eps)
        return 1 - num / (denom1 * denom2)


class MSE:
    def __call__(self, x, y, weight=None, **kwargs):
        if weight is not None:
            return (torch.square(x - y) * weight).sum() / weight.sum()
        else:
            return torch.square(x - y).mean()


class NMI(nn.Module):
    """
    Normalized mutual information, using gaussian parzen window estimates.
    Adapted from https://github.com/SteffenCzolbe/DeepSimRegistration/blob/master/src/loss_metrics.py
    """

    def __init__(self,
                 vmin=0.0,
                 vmax=1.0,
                 num_bins=64,
                 normalised=True,
                 **kwargs
                 ):
        super().__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.normalised = normalised

        # set the std of Gaussian kernel so that FWHM is one bin width
        bin_width = (vmax - vmin) / num_bins
        self.sigma = bin_width * (1 / (2 * math.sqrt(2 * math.log(2))))

        # set bin edges
        self.num_bins = num_bins
        self.bins = torch.linspace(
            self.vmin, self.vmax, self.num_bins, requires_grad=False).unsqueeze(1)

    def norm(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def _compute_joint_prob(self, x, y, w):
        """
        Compute joint distribution and entropy
        Input shapes (N, 1, prod(sizes))
        """
        # cast bins
        self.bins = self.bins.type_as(x)

        # calculate Parzen window function response (N, #bins, H*W*D)
        win_x = torch.exp(-(x - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_x = win_x / (math.sqrt(2 * math.pi) * self.sigma)
        win_y = torch.exp(-(y - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_y = win_y / (math.sqrt(2 * math.pi) * self.sigma)

        win_x = win_x * w

        # calculate joint histogram batch
        hist_joint = win_x.bmm(win_y.transpose(1, 2))  # (N, #bins, #bins)

        # normalise joint histogram to get joint distribution
        hist_norm = hist_joint.flatten(
            start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        # (N, #bins, #bins) / (N, 1, 1)
        p_joint = hist_joint / hist_norm.view(-1, 1, 1)

        return p_joint

    def __call__(self, x, y, weight=None):
        """
        Calculate (Normalised) Mutual Information Loss.
        Args:
            x: (torch.Tensor, size (N, 1, *sizes))
            y: (torch.Tensor, size (N, 1, *sizes))
        Returns:
            (Normalise)MI: (scalar)
        """

        if weight is None:
            weight = torch.ones_like(x)

        # make sure the sizes are (N, 1, prod(sizes))
        x_flat = x.flatten(start_dim=2, end_dim=-1)
        y_flat = y.flatten(start_dim=2, end_dim=-1)
        w_flat = weight.flatten(start_dim=2, end_dim=-1)

        x_flat = self.norm(x_flat)
        y_flat = self.norm(y_flat)
        w_flat = w_flat / w_flat.max()
        # compute joint distribution
        p_joint = self._compute_joint_prob(x_flat, y_flat, w_flat)

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, x bins in dim1, y bins in dim2
        p_x = torch.sum(p_joint, dim=2)
        p_y = torch.sum(p_joint, dim=1)

        # calculate entropy
        ent_x = - torch.sum(p_x * torch.log(p_x + 1e-5), dim=1)  # (N,1)
        ent_y = - torch.sum(p_y * torch.log(p_y + 1e-5), dim=1)  # (N,1)
        ent_joint = - \
            torch.sum(p_joint * torch.log(p_joint + 1e-5), dim=(1, 2))  # (N,1)

        if self.normalised:
            return -torch.mean((ent_x + ent_y) / (ent_joint + 1e-8))
        else:
            return -torch.mean(ent_x + ent_y - ent_joint)
