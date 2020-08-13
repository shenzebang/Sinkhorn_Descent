import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from matplotlib import pyplot as plt
import pickle

import os
import sys

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_path,'..'))

from core.sinkhorn_barycenter_weight import SinkhornBarycenterWeight
import numpy
from utils.plot_utils import plot

# ==============================================================================================================
#   load the file containing the ellipses support and weights
# ==============================================================================================================
save_path = 'barycenter/mnist/SD'
os.system('mkdir -p {}'.format(save_path))
dataf = 'mnist'

for digit in range(10):
    # digit = 0
    d = 2
    measures_weights = torch.load('{}/mnist_{}_weight_torch.pt'.format(dataf, digit))
    measures_locations = torch.load('{}/mnist_{}_support_torch.pt'.format(dataf, digit))
    N = len(measures_weights)
    assert len(measures_locations) == N


    source_distributions = []

    n = 100   # number of source measures
    m = 50**2             # support size of initial measures
    blur = 0.01
    p = 2
    barycenter_initial_support = torch.rand(m, d, requires_grad=True, dtype=torch.float32).cuda()
    barycenter_initial_weight = ((1 / m) * torch.ones(m, dtype=torch.float32)).cuda()

    # ==============================================================================================================
    #   compute Sinkhorn barycenter via Sinkhorn Descent
    # ==============================================================================================================

    step_size = 1  # step size
    nit = 100  # number of iterations
    frequency_loss_evaluation = 1  # the frequency to evaluate the loos
    # try:
    #     from pykeops.torch import generic_sum, generic_logsumexp
    #
    #     backend = "keops"  # Efficient GPU backend, which scales up to ~1,000,000 samples.
    # except ImportError:
    #     backend = "pytorch"  # Vanilla torch backend. Beware of memory overflows above ~10,000 samples!
    backend = 'tensorized'
    # barycenter = SinkhornBarycenter(source_distributions, barycenter_initial, step_size, backend="pytorch")
    barycenter = SinkhornBarycenterWeight(
                    source_distribution_weights = measures_weights[0:n],
                    source_distribution_supports = measures_locations[0:n],
                    barycenter_initial_weight = barycenter_initial_weight,
                    barycenter_initial_support = barycenter_initial_support,
                    step_size = step_size,
                    backend = backend,
                    blur = blur,
                    p=p
    )

    fig = plt.figure()
    ax = fig.gca()
    plt.clf()

    # ==============================================================================================================
    #   store the losses for plotting
    # ==============================================================================================================
    loss_vector = numpy.zeros(numpy.int(nit/frequency_loss_evaluation))
    iter_vector = numpy.arange(0, nit, frequency_loss_evaluation)+1

    print("...computing the Sinkhorn barycenter using Sinkhorn descent...")

    for i in range(nit):
        barycenter.step()
        if i % frequency_loss_evaluation == 0:
            barycenter_plot = barycenter.barycenter_support.detach().cpu()
            ax.scatter(barycenter_plot[:, 1], barycenter_plot[:, 0])
            plot(barycenter_plot[:, [1, 0]], barycenter_initial_weight.cpu())
            # plt.xlim(0, 1)
            # plt.ylim(0, 1)
            plt.axis('off')
            plt.savefig(os.path.join(save_path, 'sd_digit_{}_barycenter_{}.eps'.format(digit, i)), bbox_inches='tight')
            loss = barycenter.evaluate()
            loss_vector[numpy.int(i/frequency_loss_evaluation)] = loss
            print("iteration {}, loss {}".format(i, loss))
            plt.clf()
    # print(min(barycenter_plot[:, 0]), min(barycenter_plot[:, 1]), max(barycenter_plot[:, 0]), max(barycenter_plot[:, 1]))
    numpy.savetxt(os.path.join(save_path, "SD_MNIST_digit_{}_nparticle{}".format(digit, m)), numpy.vstack((iter_vector, loss_vector)))
