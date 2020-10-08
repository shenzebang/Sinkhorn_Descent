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


def ellipses(target_distributions, source_distribution, save_path, step_size=1, nit=500, frequency_loss_evaluation=10, backend="tensorized"):
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
            plt.savefig(os.path.join(save_path, 'barycenter_{}.png'.format(i)), bbox_inches='tight')
            # plt.savefig(os.path.join(save_path, 'barycenter_{}.png'.format(i)))
            # ax.clear()
            loss = sd.evaluate()
            loss_vector[np.int(i / frequency_loss_evaluation)] = loss
            print("iteration {}, loss {}".format(i, loss))


    # np.savetxt(os.path.join(plot_save_path, "SD_matching_nparticle{}".format(m)), np.vstack((iter_vector, loss_vector)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--support_size', type=int, default=320)
    parser.add_argument('--nit', type=int, default=500, help='number of SD iterations')
    parser.add_argument('--step_size', type=float, default=1, help='learning rate')

    args = parser.parse_args()

    script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_path, '../data', 'ellipses', 'Ellipses.pckl')
    save_path = os.path.join(script_path, '../out', 'ellipses')
    os.system('mkdir -p {}'.format(save_path))

    pkl_file = open(data_path, 'rb')
    D = pickle.load(pkl_file)
    pkl_file.close()
    pre_supp = D['support']
    pre_weights = D['weights']

    # create a meshgrid and interpret the image as a probability distribution on it
    X = torch.linspace(0, 1, 50)
    Y = torch.linspace(0, 1, 50)
    X, Y = torch.meshgrid(X, Y)
    X1 = X.reshape(X.shape[0] ** 2)
    Y1 = Y.reshape(Y.shape[0] ** 2)

    target_distributions = []

    for (supp_index, weights) in zip(pre_supp, pre_weights):
        support = torch.zeros((supp_index.shape[0], 2), requires_grad=False, dtype=torch.float32).cuda()
        support[:, 0] = X1[supp_index]
        support[:, 1] = Y1[supp_index]
        weights = torch.from_numpy(weights).float().cuda()
        target_distributions.append(Distribution(support, weights))

    # initialize the source_distribution
    dimension = 2
    initial_support = torch.rand(args.support_size, dimension, requires_grad=True, dtype=torch.float32).cuda()
    source_distribution = Distribution(initial_support)

    # plot_save_path = os.path.join(script_path, 'plot', 'matching')

    backend = "online"

    # image_size = [int(aspect*args.image_size), args.image_size]
    ellipses(target_distributions=target_distributions,
             source_distribution=source_distribution,
             step_size=args.step_size,
             save_path=save_path,
             backend=backend,
             nit=args.nit
             )
