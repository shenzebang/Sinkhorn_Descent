from core.sinkhorn_barycenter import SinkhornBarycenter
import torch
import os
import numpy as np
import time
n = 5  # number of source measures
m = 50000  # support size of source measures (also the initial measure)
m_initial = 5000
d = 100  # problem dimension
step_size = 1e0  # step size
nit = 100  # number of iterations
frequency_loss_evaluation = 10
try:
    from pykeops.torch import generic_sum, generic_logsumexp

    backend = "keops"  # Efficient GPU backend, which scales up to ~1,000,000 samples.
except ImportError:
    backend = "pytorch"  # Vanilla torch backend. Beware of memory overflows above ~10,000 samples!
script_path = os.path.dirname(os.path.abspath(__file__))
plot_save_path = os.path.join(script_path, 'plot', 'gaussian')

barycenter_initial = torch.rand(m_initial, d, requires_grad=True).cuda()

source_distributions = []
for i in range(n):
    mean = torch.zeros(d).cuda()
    mean[i] = 1
    source_distributions.append(mean+torch.randn(m, d).cuda())


loss_vector = np.zeros(np.int(nit/frequency_loss_evaluation))
iter_vector = np.arange(0, nit, frequency_loss_evaluation)+1
barycenter = SinkhornBarycenter(source_distributions, barycenter_initial, step_size, backend=backend)
for i in range(nit):
    if i % frequency_loss_evaluation == 0:
        loss = barycenter.evaluate()
        loss_vector[np.int(i / frequency_loss_evaluation)] = loss
        print("iteration {}, loss {}".format(i, loss))
    t1 = time.time()
    barycenter.step()
    t1 = time.time() - t1
    print('Iter:', i, '  Time:', t1)

np.savetxt(os.path.join(plot_save_path, "SD_gaussian_nparticle{}".format(m_initial)), np.vstack((iter_vector, loss_vector)))
