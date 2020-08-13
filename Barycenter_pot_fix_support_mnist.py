# -*- coding: utf-8 -*-
"""
====================================================
2D free support Wasserstein barycenters of distributions
====================================================

Illustration of 2D Wasserstein barycenters if distributions are weighted
sum of diracs.

"""

# Author: Vivien Seguy <vivien.seguy@iip.ist.i.kyoto-u.ac.jp>
#
# License: MIT License

import numpy as np
import matplotlib.pylab as pl
import ot
import torch
from matplotlib import pyplot as plt
from utils.plot_utils import plot
import os
import torch
from geomloss import SamplesLoss
##############################################################################
# Load data
# -------------
dataf = 'mnist'
save_path = 'barycenter/mnist/pot_fix_support'
os.system('mkdir -p {}'.format(save_path))
digit = 8
d = 2
n = 100
measures_weights_vec = torch.load('{}/mnist_{}_weight_vec_np.pt'.format(dataf, digit))[0:n]
measures_weights_vec_torch = torch.load('{}/mnist_{}_weight_vec_torch.pt'.format(dataf, digit))[0:n]
A = np.asarray(measures_weights_vec).transpose()


##############################################################################
# Compute free support barycenter
# -------------------------------

k = 28**2  # number of Diracs of the barycenter
X_init = np.random.normal(0., 1., (k, d))  # initial Dirac locations
b = np.ones((k,)) / k  # weights of the barycenter (it will not be optimized, only the locations are optimized)
max_it = 20
blur = 0.1
p = 2

X = torch.linspace(0, 1, 28)
Y = torch.linspace(0, 1, 28)
X, Y = torch.meshgrid(X, Y)
X_flat = torch.flatten(X)
Y_flat = torch.flatten(Y)
x_1 = torch.stack([X_flat, Y_flat], 1).numpy()
# M = ot.utils.dist0(28**2)
# M /= M.max()
M = ot.utils.dist(x_1)

support = torch.stack([X.reshape(-1), Y.reshape(-1)]).transpose_(0, 1)
support_cuda = support.cuda()
X_list = ot.bregman.barycenter_stabilized(A, M, blur**p, numItermax=max_it, save_trajecotry=True)

#########################################################################

backend = 'tensorized'
scaling = 0.95
loss_operator = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend=backend, scaling=scaling)
measures_weights_torch = torch.load('{}/mnist_{}_weight_torch.pt'.format(dataf, digit))[0:n]
measures_locations_torch = torch.load('{}/mnist_{}_support_torch.pt'.format(dataf, digit))[0:n]
fig = plt.figure()
ax = fig.gca()
weight_plot_cuda = torch.tensor(b, dtype=torch.float32).cuda()
weight_plot_cpu = torch.tensor(b, dtype=torch.float32)

for X, i in zip(X_list, range(len(X_list))):
    barycenter_weight_plot_cuda = torch.tensor(X, dtype=torch.float32).cuda()
    # compute the Sinkhorn divergence
    loss = 0
    with torch.autograd.no_grad():
        for weight in measures_weights_vec_torch:
            loss += loss_operator(barycenter_weight_plot_cuda, support_cuda, weight.cuda(), support_cuda)
    print("iteration {}, loss {}".format(i, loss))
    # plot the barycenter
    barycenter_weight_plot_cpu = barycenter_weight_plot_cuda.cpu()
    ax.scatter(support[:, 0], support[:, 1])
    plot(support[:, [1, 0]], barycenter_weight_plot_cpu)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, 'barycenter_{}.png'.format(i)), bbox_inches='tight')