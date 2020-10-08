import torch
from geomloss import SamplesLoss

class SinkhornDescent:
    def __init__(self, target_distributions, barycenter_initial, step_size, backend, blur=.05, p=2, scaling=0.95):
        self.target_distributions = target_distributions
        self.barycenter = barycenter_initial
        self.n = len(self.target_distributions)
        self.blur = blur
        self.p = p
        self.η = step_size
        self.ε = self.blur**self.p
        self.potential_operator = SamplesLoss(loss="sinkhorn", p=self.p, blur=self.blur, potentials=True, debias=False, backend=backend, scaling=scaling)
        self.loss_operator = SamplesLoss(loss="sinkhorn", p=self.p, blur=self.blur, backend=backend, scaling=scaling)
        self.backend = backend

    def step(self):
        # ==============================================================================================================
        #   compute the Sinkhorn potentials
        # ==============================================================================================================
        f_αβs = [None] * self.n
        for i in range(self.n):
            # print(i)
            f_αβs[i], _ = self.potential_operator(self.barycenter.weights, self.barycenter.support,
                                                  self.target_distributions[i].weights, self.target_distributions[i].support)
        f_αα, _ = self.potential_operator(self.barycenter.support, self.barycenter.support)
        # ==============================================================================================================
        #   compute the variation of the Sinkhorn Divergence via auto differentiation
        # ==============================================================================================================
        f_αβs_gradient = [None] * self.n
        for i in range(self.n):
            f_αβs_gradient[i] = torch.autograd.grad(torch.sum(f_αβs[i]), self.barycenter.support)[0]
        f_αα_gradient = torch.autograd.grad(torch.sum(f_αα), self.barycenter.support)[0]
        # ==============================================================================================================
        #   update the barycenter
        # ==============================================================================================================
        self.barycenter.support -= self.η * (torch.mean(torch.stack(f_αβs_gradient), dim=0) - f_αα_gradient)

    def evaluate(self):
        loss = 0
        for i in range(self.n):
            loss += self.loss_operator(self.barycenter.weights, self.barycenter.support,
                                        self.target_distributions[i].weights, self.target_distributions[i].support)
        return loss
