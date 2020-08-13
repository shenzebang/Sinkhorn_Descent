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
save_path = 'barycenter/mnist/pot_free_support'
os.system('mkdir -p {}'.format(save_path))

for digit in range(10):
    d = 2
    n = 100
    measures_weights = torch.load('{}/mnist_{}_weight_np.pt'.format(dataf, digit))[0:n]
    measures_locations = torch.load('{}/mnist_{}_support_np.pt'.format(dataf, digit))[0:n]
    # N = len(measures_weights)
    # assert len(measures_locations) == N

    ##############################################################################
    # Compute free support barycenter
    # -------------------------------

    k = 50**2  # number of Diracs of the barycenter
    X_init = np.random.normal(0., 1., (k, d))  # initial Dirac locations
    b = np.ones((k,)) / k  # weights of the barycenter (it will not be optimized, only the locations are optimized)
    max_it = 100
    X_list = ot.lp.free_support_barycenter(measures_locations, measures_weights, X_init, b, verbose=True, numItermax=max_it, save_trajecotry=True, stopThr=0)


    ##############################################################################
    blur = 0.01
    p = 2
    backend = 'tensorized'
    scaling = 0.95
    loss_operator = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend=backend, scaling=scaling)
    measures_weights_torch = torch.load('{}/mnist_{}_weight_torch.pt'.format(dataf, digit))[0:n]
    measures_locations_torch = torch.load('{}/mnist_{}_support_torch.pt'.format(dataf, digit))[0:n]
    fig = plt.figure()
    ax = fig.gca()
    plt.clf()
    weight_plot_cuda = torch.tensor(b, dtype=torch.float32).cuda()
    weight_plot_cpu = torch.tensor(b, dtype=torch.float32)


    loss_vector = np.zeros(max_it)
    iter_vector = np.arange(0, max_it, 1)+1
    for X, i in zip(X_list, range(len(X_list))):
        barycenter_plot = torch.tensor(X, dtype=torch.float32).cuda()
        # compute the Sinkhorn divergence
        loss = 0
        with torch.autograd.no_grad():
            for weight, support in zip(measures_weights_torch, measures_locations_torch):
                loss += loss_operator(weight_plot_cuda, barycenter_plot, weight.cuda(), support.cuda())
        print("iteration {}, loss {}".format(i, loss))
        # plot the barycenter
        barycenter_plot = barycenter_plot.cpu()
        ax.scatter(barycenter_plot[:, 0], barycenter_plot[:, 1])
        plot(barycenter_plot[:, [1, 0]], weight_plot_cpu)
        plt.axis('off')
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        plt.savefig(os.path.join(save_path, 'fs_digit_{}_barycenter_{}.eps'.format(digit, i)), bbox_inches='tight')
        plt.clf()
        loss_vector[np.int(i)] = loss

    np.savetxt(os.path.join(save_path, "SD_MNIST_digit_{}_nparticle{}".format(digit, k)),
                      np.vstack((iter_vector, loss_vector)))
