# import torch
# from torch import norm
# from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
# from torch import sigmoid
# from torch.nn import MSELoss
#
#
# def squared_distances(x, y):
#     if x.dim() == 2:
#         D_xx = (x * x).sum(-1).unsqueeze(1)  # (N,1)
#         D_xy = torch.matmul(x, y.permute(1, 0))  # (N,D) @ (D,M) = (N,M)
#         D_yy = (y * y).sum(-1).unsqueeze(0)  # (1,M)
#     elif x.dim() == 3:  # Batch computation
#         D_xx = (x * x).sum(-1).unsqueeze(2)  # (B,N,1)
#         D_xy = torch.matmul(x, y.permute(0, 2, 1))  # (B,N,D) @ (B,D,M) = (B,N,M)
#         D_yy = (y * y).sum(-1).unsqueeze(1)  # (B,1,M)
#     else:
#         print("x.shape : ", x.shape)
#         raise ValueError("Incorrect number of dimensions")
#
#     return D_xx - 2 * D_xy + D_yy
#
# class GroundCost(torch.nn.Module):
#     def __init__(self, D_ground, D_hidden, D_feature):
#         super(GroundCost, self).__init__()
#         self.linear1 = torch.nn.Linear(D_ground, D_hidden).cuda()
#         self.linear2 = torch.nn.Linear(D_hidden, D_feature).cuda()
#
#     def forward(self, x, y):
#         x_sigmoid = sigmoid(self.linear1(x))
#         x_feature = self.linear2(x_sigmoid)
#
#         y_sigmoid = sigmoid(self.linear1(y))
#         y_feature = self.linear2(y_sigmoid)
#
#         feature_cost = squared_distances(x_feature, y_feature)
#         return feature_cost
#
#
#
#
# # Create some large point clouds in 3D
# x = torch.randn(100, 3, requires_grad=True).cuda()
# y = torch.randn(100, 3, requires_grad=True).cuda()
# ground_cost = GroundCost(3, 3, 3)
# Define a Sinkhorn (~Wasserstein) loss between sampled measures
# ==================================================================
# loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05, debias=False, cost=ground_cost, backend="tensorized", scaling=.9)
# loss_xx = loss(x, x)
# grad_c = torch.autograd.grad(loss_xx, ground_cost.parameters())[0]
# print(torch.norm(grad_c))
# ==================================================================

# ==================================================================
# loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05, debias=False, backend="tensorized", scaling=.9)
# loss_xx = loss(x, x)
# loss_xy = loss(x, y)
# print(loss_xy - loss_xx)
# ==================================================================
#
# loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
# L = loss(x, y)*100000
# print(L)
#
# loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05, potentials=True, debias=False)
# F_xy, G_xy = loss(x, y)
# F_xx, G_xx = loss(x, x)
# F_yy, G_yy = loss(y, y)
# print(norm(F_xx - G_xx)/norm(F_xx))
# print(norm(F_yy - G_yy)/norm(F_yy))
#
# F_ = F_xy - F_xx
# G_ = G_xy - G_yy
# print(torch.norm(F - F_)/torch.norm(F))
# print(torch.norm(G - G_)/torch.norm(G))
#
# # g_x, = torch.autograd.grad(L, [x])  # GeomLoss fully supports autograd!
# # g_y, = torch.autograd.grad(L, [y])



#
from pykeops.torch import LazyTensor
import torch

use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

M = 1000
D = 10
x = torch.randn(M, D).type(tensor)  # M target points in dimension D, stored on the GPU
# y = torch.randn(N, D).type(tensor)  # N source points in dimension D, stored on the GPU
b = torch.randn(M, 4).type(tensor)  # N values of the 4D source signal, stored on the GPU


x_i = LazyTensor(x[:, None, :])  # (M, 1, D) LazyTensor
y_j = LazyTensor(x[None, :, :])  # (1, N, D) LazyTensor

D_ij = ((x_i - y_j) ** 2).sum(-1).sqrt()  # Symbolic (M, N) matrix of distances
K_ij = (- D_ij).exp()  # Symbolic (M, N) Laplacian (aka. exponential) kernel matrix
a_i = K_ij @ b  # The matrix-vector product "@" can be used on "raw" PyTorch tensors!

print("a_i is now a {} of shape {}.".format(type(a_i), a_i.shape))

#

