import torch
from geomloss import SamplesLoss

class SinkhornBarycenter:
    def __init__(self, source_distributions, barycenter_initial, step_size, backend, blur=.05, p=2):
        self.source_distributions = source_distributions
        self.barycenter = barycenter_initial
        self.n = len(self.source_distributions)
        self.blur = blur
        self.p = p
        self.η = step_size
        self.ε = self.blur**self.p
        # self.f_αβs_gradient = [None] * len(self.source_distributions)
        # self.f_αα_gradient = None
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
        f_αβs = [None] * len(self.source_distributions)
        for i in range(self.n):
            f_αβs[i], _ = self.potential_operator(self.barycenter, self.source_distributions[i])
        f_αα, _ = self.potential_operator(self.barycenter, self.barycenter)
        # ==============================================================================================================
        #   compute the variation of the Sinkhorn Divergence via auto differentiation
        # ==============================================================================================================
        f_αβs_gradient = [None] * len(self.source_distributions)
        for i in range(self.n):
            f_αβs_gradient[i] = torch.autograd.grad(torch.sum(f_αβs[i]), self.barycenter)[0]
        f_αα_gradient = torch.autograd.grad(torch.sum(f_αα), self.barycenter)[0]
        # ==============================================================================================================
        #   update the barycenter
        # ==============================================================================================================
        self.barycenter -= self.η * (torch.mean(torch.stack(f_αβs_gradient), dim=0) - f_αα_gradient)

    def evaluate(self):
        loss = 0
        for i in range(self.n):
            loss += self.loss_operator(self.barycenter, self.source_distributions[i])
        return loss
