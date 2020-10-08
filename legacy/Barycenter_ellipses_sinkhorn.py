import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from matplotlib import pyplot as plt
import pickle

import os
import sys

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_path, '../..'))

from core.sinkhorn_barycenter import SinkhornBarycenter
import numpy
from utils.plot_utils import plot

data_path = os.path.join(script_path, '../data', 'ellipses', 'Ellipses.pckl')
save_path = os.path.join(script_path, '../out', 'ellipses')
plot_save_path = os.path.join(script_path, 'plot', 'ellipses')
os.system('mkdir -p {}'.format(save_path))

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

source_distributions = []

n = len(pre_supp)   # number of source measures
m = 320             # support size of initial measures
d = 2
for i in range(n):
    supp = torch.zeros((pre_supp[i].shape[0], 2), requires_grad=True, dtype=torch.float32).cuda()
    supp[:, 0] = X1[pre_supp[i]]
    supp[:, 1] = Y1[pre_supp[i]]
    weights = (1 / pre_supp[i].shape[0]) * torch.ones(pre_supp[i].shape[0], 1)
    source_distributions.append(supp)

barycenter_initial = torch.rand(m, d, requires_grad=True, dtype=torch.float32).cuda()

# ==============================================================================================================
#   compute Sinkhorn barycenter via Sinkhorn Descent
# ==============================================================================================================

step_size = 1  # step size
nit = 12  # number of iterations
frequency_loss_evaluation = 2  # the frequency to evaluate the loos
try:
    from pykeops.torch import generic_sum, generic_logsumexp

    backend = "keops"  # Efficient GPU backend, which scales up to ~1,000,000 samples.
except ImportError:
    backend = "pytorch"  # Vanilla torch backend. Beware of memory overflows above ~10,000 samples!

# barycenter = SinkhornBarycenter(source_distributions, barycenter_initial, step_size, backend="pytorch")
barycenter = SinkhornBarycenter(source_distributions, barycenter_initial, step_size, backend=backend, blur=0.01)
barycenter_weight = (1 / m) * torch.ones(m, dtype=torch.float32)

fig = plt.figure()
ax = fig.gca()

# ==============================================================================================================
#   store the losses for plotting
# ==============================================================================================================
loss_vector = numpy.zeros(numpy.int(nit/frequency_loss_evaluation))
iter_vector = numpy.arange(0, nit, frequency_loss_evaluation)+1


for i in range(nit):
    barycenter.step()
    if i % frequency_loss_evaluation == 0:
        barycenter_plot = torch.Tensor.cpu(barycenter.barycenter).detach().numpy()
        # ax.scatter(barycenter_plot[:, 0], barycenter_plot[:, 1])
        plot(barycenter_plot, barycenter_weight)
        plt.axis('off')
        plt.savefig(os.path.join(save_path, 'barycenter_{}.png'.format(i)), bbox_inches='tight')
        loss = barycenter.evaluate()
        loss_vector[numpy.int(i/frequency_loss_evaluation)] = loss
        print("iteration {}, loss {}".format(i, loss))

print(min(barycenter_plot[:, 0]), min(barycenter_plot[:, 1]), max(barycenter_plot[:, 0]), max(barycenter_plot[:, 1]))
# numpy.savetxt(os.path.join(plot_save_path, "SD_ellipses_nparticle{}".format(m)), numpy.vstack((iter_vector, loss_vector)))
