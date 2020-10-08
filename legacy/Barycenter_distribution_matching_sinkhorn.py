import torch
# torch.set_default_tensor_type(torch.DoubleTensor)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from PIL import Image

import numpy as np

import os
import sys
from core.sinkhorn_barycenter_weight import SinkhornBarycenterWeight
from utils.plot_utils import plot
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_path, '../..'))


data_path = os.path.join(script_path, '../data', 'matching', 'cheetah.jpg')
save_path = os.path.join(script_path, '../out', 'matching')
plot_save_path = os.path.join(script_path, 'plot', 'matching')
os.system('mkdir -p {}'.format(save_path))


im_size = 100

# load and resize image
img = Image.open(data_path)
img.thumbnail((im_size,im_size), Image.ANTIALIAS)  # resizes image in-place
# imgplot = plt.imshow(img)

pix = np.array(img)
min_side = np.min(pix[:, :, 0].shape)
pix = 255 - pix[0:min_side, 0:min_side]


# visualize and save the resized original image
try:
    imgplot = plt.imshow(img)
    plt.savefig(os.path.join(save_path,'original.png'))
    plt.pause(0.1)
finally:
    pass

# print(min(pix[:, 0]), min(pix[:, 1]), max(pix[:, 0]), max(pix[:, 1]))


# create a meshgrid and interpret the image as a probability distribution on it
x = torch.linspace(0, 1, steps=pix.shape[0])
y = torch.linspace(0, 1, steps=pix.shape[0])
X, Y = torch.meshgrid(x, y)
X1 = X.reshape(X.shape[0] ** 2)
Y1 = Y.reshape(Y.shape[0] ** 2)
n = X.shape[0] ** 2
y1 = []

MX = max(X1)

weights = []
pix_arr = pix[:, :, 0].reshape(pix.shape[0] ** 2)
for i in range(n):
    if pix_arr[i] > 50:
        y1.append(torch.tensor([Y1[i], MX - X1[i]]))
        weights.append(torch.tensor(pix_arr[i], dtype=torch.float32))

nu1t = torch.stack(y1)
w1 = torch.stack(weights).reshape((len(weights), 1))
w1 = w1 / (torch.sum(w1, dim=0)[0])
support_meas = nu1t
weights_meas = w1

source_distribution_supports = []
source_distribution_weights = []
m = 8000             # support size of initial measures
d = 2
barycenter_weight = (1 / m) * torch.ones(m, dtype=torch.float32)

source_distribution_supports.append(support_meas.float().cuda())
source_distribution_weights.append(weights_meas.float().cuda().squeeze())

barycenter_initial_support = torch.rand(m, d, requires_grad=True, dtype=torch.float32).cuda()
barycenter_initial_weight = ((1 / barycenter_initial_support.shape[0]) * torch.ones(barycenter_initial_support.shape[0], dtype=torch.float32)).cuda()

# ==============================================================================================================
#   compute Sinkhorn barycenter via Sinkhorn Descent
# ==============================================================================================================

step_size = 1e-0  # step size
nit = 201  # number of iterations
frequency_loss_evaluation = 20
try:
    from pykeops.torch import generic_sum, generic_logsumexp

    backend = "keops"  # Efficient GPU backend, which scales up to ~1,000,000 samples.
except ImportError:
    backend = "pytorch"  # Vanilla torch backend. Beware of memory overflows above ~10,000 samples!

# barycenter = SinkhornBarycenter(source_distributions, barycenter_initial, step_size, backend="pytorch")
barycenter = SinkhornBarycenterWeight(source_distribution_weights, source_distribution_supports, barycenter_initial_weight, barycenter_initial_support, step_size, backend=backend, blur=0.01)
fig = plt.figure()
ax = fig.gca()

# ==============================================================================================================
#   store the losses for plotting
# ==============================================================================================================
loss_vector = np.zeros(np.int(nit/frequency_loss_evaluation)+1)
iter_vector = np.arange(0, nit, frequency_loss_evaluation)+1

for i in range(nit):
    barycenter.step()
    if i % frequency_loss_evaluation == 0:
        barycenter_plot = torch.Tensor.cpu(barycenter.barycenter_support).detach().numpy()
        # ax.scatter(barycenter_plot[:, 0], barycenter_plot[:, 1])
        plot(barycenter_plot, barycenter_weight, bins=im_size)
        plt.axis('off')
        plt.savefig(os.path.join(save_path, 'barycenter_{}.png'.format(i)), bbox_inches='tight')
        # ax.clear()
        loss = barycenter.evaluate()
        loss_vector[np.int(i / frequency_loss_evaluation)] = loss
        print("iteration {}, loss {}".format(i, loss))

print(min(barycenter_plot[:, 0]), min(barycenter_plot[:, 1]), max(barycenter_plot[:, 0]), max(barycenter_plot[:, 1]))

# np.savetxt(os.path.join(plot_save_path, "SD_matching_nparticle{}".format(m)), np.vstack((iter_vector, loss_vector)))
