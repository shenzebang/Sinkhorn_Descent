import torch
from torch.nn import Module
import torch.optim as optim


class EntropyRegularizedOT(Module):
    """
    A class that computes the Sinkhorn potentials to the entropy regularized optimal transport problem between two
    probability measures α and β.
    Note:
        1.  α should be a discrete measure
        2.  A subroutine for generating samples according to β should be provided.
        3.  In such setting, this class computes the entropy regularized optimal transport distance between α and β
            using the stochastic gradient type methods based on the paper "Stochastic Optimization for Large-scale
            Optimal Transport".
        4.  **cost** should be a python function that takes as input a (B,N,D) torch Tensor **x**,
            a (B,M,D) torch Tensor **y** and returns a batched Cost matrix as a (B,N,M) Tensor.
    :parameter

    """
    def __init__(self, cost, gamma=1e-3, lr=1e-4, n_loops=1e3):
        super(EntropyRegularizedOT, self).__init__()
        self.gamma = gamma
        self.cost = cost
        self.lr = lr
        self.n_loops = n_loops

    def forward(self, α, β_sampler):
        """
        Computes the Sinkhorn potential between α and β
        :param α:           a discrete probability measure instance
        :param β_sampler:   a sampler that generates samples from β. β_sampler.sample() should output a discrete
                            probability measure instance
        :return:
        """
        N = α.support.shape[0]
        f = torch.ones(N).type_as(α.support)
        for l in range(self.n_loops):
            β_sample = β_sampler.sample()




            C_xx, C_yy = (self.cost(x, x.detach()), self.cost(y, y.detach())) if self.debias else (None, None)
            C_xy, C_yx = (self.cost(x, y.detach()), self.cost(y, x.detach()))

        a_x, b_y, a_y, b_x = sinkhorn_loop(softmin_tensorized,
                                           log_weights(α), log_weights(β),
                                           C_xx, C_yy, C_xy, C_yx, ε_s, ρ, debias=debias)

        return sinkhorn_cost(ε, ρ, α, β, a_x, b_y, a_y, b_x, batch=True, debias=debias, potentials=potentials)


    def softmin_tensorized(self, ε, C, f):
        B = C.shape[0]
        return - ε * (f.view(B, 1, -1) - C / ε).logsumexp(2).view(-1)

    def process_args(self, *args):
        if len(args) == 4:
            α, x, β, y = args
            return None, α, x, None, β, y
        elif len(args) == 2:
            x, y = args
            α = self.generate_weights(x)
            β = self.generate_weights(y)
            return None, α, x, None, β, y
        else:
            raise ValueError("A SamplesLoss accepts two (x, y), four (α, x, β, y) arguments.")

    def generate_weights(self, x):
        N = x.shape[0]
        return torch.ones(N).type_as(x) / N

    def check_shapes(self, l_x, α, x, l_y, β, y):

        if α.dim() != β.dim(): raise ValueError("Input weights 'α' and 'β' should have the same number of dimensions.")
        if x.dim() != y.dim(): raise ValueError("Input samples 'x' and 'y' should have the same number of dimensions.")
        if x.shape[-1] != y.shape[-1]: raise ValueError(
            "Input samples 'x' and 'y' should have the same last dimension.")

        if x.dim() == 2:  # No batch --------------------------------------------------------------------
            B = 0  # Batchsize
            N, D = x.shape  # Number of "i" samples, dimension of the feature space
            M, _ = y.shape  # Number of "j" samples, dimension of the feature space

            if α.dim() not in [1, 2]:
                raise ValueError(
                    "Without batches, input weights 'α' and 'β' should be encoded as (N,) or (N,1) tensors.")
            elif α.dim() == 2:
                if α.shape[1] > 1: raise ValueError(
                    "Without batches, input weights 'α' should be encoded as (N,) or (N,1) tensors.")
                if β.shape[1] > 1: raise ValueError(
                    "Without batches, input weights 'β' should be encoded as (M,) or (M,1) tensors.")
                α, β = α.view(-1), β.view(-1)

            if l_x is not None:
                if l_x.dim() not in [1, 2]:
                    raise ValueError(
                        "Without batches, the vector of labels 'l_x' should be encoded as an (N,) or (N,1) tensor.")
                elif l_x.dim() == 2:
                    if l_x.shape[1] > 1: raise ValueError(
                        "Without batches, the vector of labels 'l_x' should be encoded as (N,) or (N,1) tensors.")
                    l_x = l_x.view(-1)
                if len(l_x) != N: raise ValueError(
                    "The vector of labels 'l_x' should have the same length as the point cloud 'x'.")

            if l_y is not None:
                if l_y.dim() not in [1, 2]:
                    raise ValueError(
                        "Without batches, the vector of labels 'l_y' should be encoded as an (M,) or (M,1) tensor.")
                elif l_y.dim() == 2:
                    if l_y.shape[1] > 1: raise ValueError(
                        "Without batches, the vector of labels 'l_y' should be encoded as (M,) or (M,1) tensors.")
                    l_y = l_y.view(-1)
                if len(l_y) != M: raise ValueError(
                    "The vector of labels 'l_y' should have the same length as the point cloud 'y'.")

            N2, M2 = α.shape[0], β.shape[0]

        elif x.dim() == 3:  # batch computation ---------------------------------------------------------
            B, N, D = x.shape  # Batchsize, number of "i" samples, dimension of the feature space
            B2, M, _ = y.shape  # Batchsize, number of "j" samples, dimension of the feature space
            if B != B2: raise ValueError("Samples 'x' and 'y' should have the same batchsize.")

            if α.dim() not in [2, 3]:
                raise ValueError(
                    "With batches, input weights 'α' and 'β' should be encoded as (B,N) or (B,N,1) tensors.")
            elif α.dim() == 3:
                if α.shape[2] > 1: raise ValueError(
                    "With batches, input weights 'α' should be encoded as (B,N) or (B,N,1) tensors.")
                if β.shape[2] > 1: raise ValueError(
                    "With batches, input weights 'β' should be encoded as (B,M) or (B,M,1) tensors.")
                α, β = α.squeeze(-1), β.squeeze(-1)

            if l_x is not None: raise NotImplementedError(
                'The "multiscale" backend has not been implemented with batches.')
            if l_y is not None: raise NotImplementedError(
                'The "multiscale" backend has not been implemented with batches.')

            B2, N2 = α.shape
            B3, M2 = β.shape
            if B != B2: raise ValueError("Samples 'x' and weights 'α' should have the same batchsize.")
            if B != B3: raise ValueError("Samples 'y' and weights 'β' should have the same batchsize.")

        else:
            raise ValueError("Input samples 'x' and 'y' should be encoded as (N,D) or (B,N,D) (batch) tensors.")

        if N != N2: raise ValueError("Weights 'α' and samples 'x' should have compatible shapes.")
        if M != M2: raise ValueError("Weights 'β' and samples 'y' should have compatible shapes.")

        return B, N, M, D