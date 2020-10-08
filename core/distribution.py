import torch

class Distribution:
    """
        Parameters:
            support (torch.tensor): The support of the distribution. [N, d]

            weights (torch.tensor): The weights on the support of the distribution. [N]
    """
    def __init__(self, support, weights=None):
        if weights is not None:
            if support.shape[0] != weights.shape[0]:
                raise ValueError("Input support and weights should have the same number of entries")
        else:
            weights = (1 / support.shape[0]) * torch.ones(support.shape[0], dtype=support.dtype).cuda()
        self.support = support
        self.weights = weights
        self.dimension = support.shape[1]

    def to_cuda(self):
        self.support = self.support.cuda()
        self.weights = self.weights.cuda()

