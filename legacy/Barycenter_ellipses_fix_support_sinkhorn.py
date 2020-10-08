import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import pickle

import os
import sys
import numpy
from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_path, '../..'))

from core.fix_support_sinkhorn_barycenter import FixSupportSinkhornBarycenter


problem_name = 'ellipses'
algorithm_name = 'fix_support'

data_path = os.path.join(script_path, 'data/ellipses', 'Ellipses.pckl')
save_path = os.path.join(script_path, 'output/barycenter/ellipses', 'ellipses')
plot_save_path = os.path.join(script_path, 'plot/barycenter/ellipses/fix_support', 'ellipses')

# ==============================================================================================================
#   load the file containing the ellipses support and weights
# ==============================================================================================================
pkl_file = open(data_path, 'rb')
D = pickle.load(pkl_file)
pkl_file.close()
pre_supp = D['support']
pre_weights = D['weights']
scale = 1
X = torch.linspace(0, scale * 1, 50)
Y = torch.linspace(0, scale * 1, 50)
X, Y = torch.meshgrid(X, Y)
X1 = X.reshape(X.shape[0] ** 2)
Y1 = Y.reshape(Y.shape[0] ** 2)

source_distributions_supports = []
source_distributions_weights = []
n = len(pre_supp)   # number of source measures
m = 600             # support size of initial measures
d = 2
frequency_loss_evaluation = 10
for i in range(n):
    supp = torch.zeros((pre_supp[i].shape[0], 2), dtype=torch.float32).cuda()
    supp[:, 0] = X1[pre_supp[i]]
    supp[:, 1] = Y1[pre_supp[i]]
    weights = ((1 / pre_supp[i].shape[0]) * torch.ones(pre_supp[i].shape[0], dtype=torch.float32)).cuda()
    source_distributions_supports.append(supp)
    source_distributions_weights.append(weights)

# ==============================================================================================================
#   compute Sinkhorn barycenter via Sinkhorn Descent
# ==============================================================================================================

step_size = 1e0  # step size
nit = 500  # number of iterations

try:
    from pykeops.torch import generic_sum, generic_logsumexp

    backend = "keops"  # Efficient GPU backend, which scales up to ~1,000,000 samples.
except ImportError:
    backend = "pytorch"  # Vanilla torch backend. Beware of memory overflows above ~10,000 samples!

barycenter = FixSupportSinkhornBarycenter(source_distributions_supports, source_distributions_weights,
                        step_size, backend=backend)  # support and weights of the barycenter is automatically determined
fig = plt.figure()
ax = fig.gca()
loss_vector = numpy.zeros(numpy.int(nit/frequency_loss_evaluation))
iter_vector = numpy.arange(0, nit, frequency_loss_evaluation)+1

# fig = plt.figure()
# ax = fig.gca()
for i in range(nit):
    loss = barycenter.step()
    if i % frequency_loss_evaluation == 0:
        # barycenter_plot = torch.Tensor.cpu(barycenter.barycenter_support).detach().numpy()
        # ax.scatter(barycenter_plot[:, 0], barycenter_plot[:, 1])
        # plt.savefig(os.path.join(save_path, 'barycenter_{}.png'.format(i)))
        # ax.clear()
        loss = barycenter.evaluate()
        loss_vector[numpy.int(i / frequency_loss_evaluation)] = loss
        print("iteration {}, loss {}".format(i, loss))

numpy.savetxt(os.path.join(plot_save_path, "FS_ellipses_nparticle{}".format(m)), numpy.vstack((iter_vector, loss_vector)))
