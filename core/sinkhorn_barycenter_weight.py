import torch
from geomloss import SamplesLoss


class SinkhornBarycenterWeight:
    def __init__(self, source_distribution_weights, source_distribution_supports, barycenter_initial_weight,
                 barycenter_initial_support, step_size, backend, blur=.05, p=2):
        self.source_distribution_weights = source_distribution_weights
        self.source_distribution_supports = source_distribution_supports
        self.barycenter_support = barycenter_initial_support
        self.barycenter_weight = barycenter_initial_weight
        self.n = len(self.source_distribution_weights)
        self.blur = blur
        self.p = p
        self.η = step_size
        self.ε = self.blur**self.p
        if backend == "keops":
            sample_loss_backend = "online"
        else:
            sample_loss_backend = "auto"
        self.potential_operator = SamplesLoss(loss="sinkhorn", p=self.p, blur=self.blur, potentials=True, debias=False, backend=sample_loss_backend)
        self.loss_operator = SamplesLoss(loss="sinkhorn", p=self.p, blur=self.blur, backend=sample_loss_backend)
        self.backend = backend

    def step(self):
        # ==============================================================================================================
        #   compute the Sinkhorn potentials
        # ==============================================================================================================
        f_αβs = [None] * self.n
        for i in range(self.n):
            f_αβs[i], _ = self.potential_operator(self.barycenter_weight, self.barycenter_support,
                                                  self.source_distribution_weights[i], self.source_distribution_supports[i])
        f_αα, _ = self.potential_operator(self.barycenter_support, self.barycenter_support)
        # ==============================================================================================================
        #   compute the variation of the Sinkhorn Divergence via auto differentiation
        # ==============================================================================================================
        f_αβs_gradient = [None] * self.n
        for i in range(self.n):
            f_αβs_gradient[i] = torch.autograd.grad(torch.sum(f_αβs[i]), self.barycenter_support)[0]
        f_αα_gradient = torch.autograd.grad(torch.sum(f_αα), self.barycenter_support)[0]
        # ==============================================================================================================
        #   update the barycenter
        # ==============================================================================================================
        self.barycenter_support -= self.η * (torch.mean(torch.stack(f_αβs_gradient), dim=0) - f_αα_gradient)

    def evaluate(self):
        loss = 0
        for i in range(self.n):
            loss += self.loss_operator(self.barycenter_weight, self.barycenter_support,
                                        self.source_distribution_weights[i], self.source_distribution_supports[i])
        return loss
