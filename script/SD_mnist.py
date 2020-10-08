import argparse
import torch
# torch.set_default_tensor_type(torch.DoubleTensor)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle

import numpy as np

import os
from core.sinkhorn_descent import SinkhornDescent
from core.distribution import Distribution
from utils.plot_utils import plot


def mnist(target_distributions, source_distribution, save_path, step_size=1, nit=500, frequency_loss_evaluation=1, backend="tensorized"):
    # ==============================================================================================================
    #   compute Sinkhorn barycenter via Sinkhorn Descent
    # ==============================================================================================================
    sd = SinkhornDescent(target_distributions, source_distribution, step_size, backend=backend, blur=0.01)
    fig = plt.figure()
    ax = fig.gca()
    plt.clf()
    # ==============================================================================================================
    #   store the losses for plotting
    # ==============================================================================================================
    loss_vector = np.zeros(np.int(nit/frequency_loss_evaluation)+1)
    iter_vector = np.arange(0, nit, frequency_loss_evaluation)+1

    for i in range(nit):
        sd.step()
        if i % frequency_loss_evaluation == 0:
            barycenter_plot = sd.barycenter.support.detach().cpu()
            barycenter_weight = sd.barycenter.weights.cpu()
            plot(barycenter_plot, barycenter_weight, bins=100)
            # plt.axis('off')
            plt.pause(1e-13)
            plt.savefig(os.path.join(save_path, 'barycenter_{}.png'.format(i)), bbox_inches='tight')
            # plt.savefig(os.path.join(save_path, 'barycenter_{}.png'.format(i)))
            # ax.clear()
            loss = sd.evaluate()
            loss_vector[np.int(i / frequency_loss_evaluation)] = loss
            print("iteration {}, loss {}".format(i, loss))
            plt.clf()


    # np.savetxt(os.path.join(plot_save_path, "SD_matching_nparticle{}".format(m)), np.vstack((iter_vector, loss_vector)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--support_size', type=int, default=2500)
    parser.add_argument('--batch_size', type=int, default=100, help='number of samples per digit')
    parser.add_argument('--nit', type=int, default=100, help='number of SD iterations')
    parser.add_argument('--step_size', type=float, default=1, help='learning rate')

    args = parser.parse_args()

    script_path = os.path.dirname(os.path.abspath(__file__))

    for digit in range(10):
        data_path = os.path.join(script_path, '../data', 'mnist', 'digit_{}.pt'.format(digit))
        save_path = os.path.join(script_path, '../out', 'mnist', 'digit_{}'.format(digit))
        os.system('mkdir -p {}'.format(save_path))

        # moving data to cuda
        target_distributions = torch.load(data_path)[0: args.batch_size]
        for target_distribution in target_distributions:
            target_distribution.to_cuda()

        # initialize the source_distribution
        dimension = 2
        initial_support = torch.rand(args.support_size, dimension, requires_grad=True, dtype=torch.float32).cuda()
        source_distribution = Distribution(initial_support)

        # plot_save_path = os.path.join(script_path, 'plot', 'matching')

        backend = "tensorized"

        # image_size = [int(aspect*args.image_size), args.image_size]
        mnist(target_distributions=target_distributions,
              source_distribution=source_distribution,
              step_size=args.step_size,
              save_path=save_path,
              backend=backend,
              nit=args.nit
              )
