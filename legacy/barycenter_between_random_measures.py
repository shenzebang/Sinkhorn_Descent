from core.sinkhorn_barycenter import SinkhornBarycenter
import torch

n = 10  # number of source measures
m = 100000  # support size of source measures (also the initial measure)
d = 40  # problem dimension
step_size = 1e-2  # step size
nit = 1000  # number of iterations

try:
    from pykeops.torch import generic_sum, generic_logsumexp

    backend = "keops"  # Efficient GPU backend, which scales up to ~1,000,000 samples.
except ImportError:
    backend = "pytorch"  # Vanilla torch backend. Beware of memory overflows above ~10,000 samples!

barycenter_initial = torch.randn(m, d, requires_grad=True).cuda()
source_distributions = []
for i in range(n):
    source_distributions.append(torch.randn(m, d, requires_grad=True).cuda())

# barycenter = SinkhornBarycenter(source_distributions, barycenter_initial, step_size, backend="pytorch")
barycenter = SinkhornBarycenter(source_distributions, barycenter_initial, step_size, backend=backend)
for i in range(nit):
    barycenter.step()
    loss = barycenter.evaluate()
    print("iteration {}, loss {}".format(i, loss))
