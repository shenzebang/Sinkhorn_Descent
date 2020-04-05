import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from geomloss import SamplesLoss
from utils import base_module
import time
from pykeops.torch import LazyTensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outf', default='tmp/celebrity', help='where to save results')
    parser.add_argument('--modelf', default='model/celebrity', help='where to save cost model')
    parser.add_argument('--modelp', default='particle/celebrity', help='where to save particles')
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--n_particles', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--niterD', type=int, default=0, help='no. updates of D per update of G')
    parser.add_argument('--niterG', type=int, default=1, help='no. updates of G per update of D')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.0, help='Lagrange multiplier')
    parser.add_argument('--rho', type=float, default=1e-5, help='quadratic weight penalty')
    args = parser.parse_args()

    cudnn.benchmark = True

    os.system('mkdir -p {}'.format(args.outf))
    os.system('mkdir -p {}'.format(args.modelf))
    os.system('mkdir -p {}'.format(args.modelp))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_SIZE = 32
    n_channels = 3
    η_c = 5e-1
    η_g = 1
    p = 2
    blur = 5
    scaling = .95
    d = n_channels*IMAGE_SIZE*IMAGE_SIZE
    d_feature = 32
    diameter = 32
    kernel_parameter = 1
    noise_level = 1/10
    c_shape = [-1, n_channels, IMAGE_SIZE, IMAGE_SIZE]
    dataset = dset.CelebA(root='celeba', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(IMAGE_SIZE),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    img, lab = dataset.__getitem__(0)
    _, IMAGE_SIZE_1, IMAGE_SIZE_2 = img.shape
    # print(img.shape)
    assert(IMAGE_SIZE_2 == IMAGE_SIZE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=8, drop_last=True)

    # data = next(iter(dataloader))
    # print(data[0].shape)
    # random particles as the initial measure
    μ = torch.rand(args.n_particles, n_channels, IMAGE_SIZE_1, IMAGE_SIZE_2).cuda()
    μ = μ * 2 - 1
    μ.requires_grad_()

    μ_weight = torch.ones(args.n_particles, requires_grad=False).type_as(μ) / args.n_particles

    x_real_weight = torch.ones(args.batch_size, requires_grad=False).type_as(μ) / args.batch_size

    c_encoder = base_module.Encoder(IMAGE_SIZE, n_channels, k=d_feature).cuda()

    c_encoder.load_state_dict(torch.load('{}/c_encoder_{}.pt'.format(args.modelf, 202)))

    optimizerC = optim.Adam(c_encoder.parameters(), lr=args.lr, betas=(0.5, 0.9), amsgrad=True)
    # optimizerC = optim.SGD(c_encoder.parameters(), lr=η_c)

    # losses = []
    # TODO: resolve the performance loss due to the call to max_diameter
    losses = []
    sinkhorn_divergence = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend="online", scaling=scaling)
    potential_operator = SamplesLoss(loss="sinkhorn", p=p, blur=blur, potentials=True, debias=True, backend="online",
                                     scaling=scaling)

    for epoch in range(args.epochs):
        time_start = time.time()
        loss = 0
        for i, data in enumerate(dataloader):
            x_real = data[0].to(device)
            with torch.autograd.no_grad():
                φ_x_real = c_encoder(x_real).view(-1, d_feature)

            φ_μ = c_encoder(μ).squeeze()
            # print(φ_μ.shape)
            # print(φ_x_real.shape)
            f_αβ_f_αα, g_αβ_g_αα = potential_operator(μ_weight, φ_μ, x_real_weight, φ_x_real)
            f_αβ_f_αα_gradient = torch.autograd.grad(torch.sum(f_αβ_f_αα), μ)[0]

            μ = μ - η_g * f_αβ_f_αα_gradient
            with torch.autograd.no_grad():
                loss += torch.sum(f_αβ_f_αα).item()/args.n_particles + torch.sum(g_αβ_g_αα).item()/args.batch_size
            del φ_μ, f_αβ_f_αα, f_αβ_f_αα_gradient

        loss = loss/(i+1)
        losses.append(loss)
        print("epoch {0}, μ loss {1:9.3e}, time {2}".format(epoch, loss, time.time()-time_start))
        # generated images and loss curve
        vutils.save_image(μ, '{}/x_{}.png'.format(args.outf, epoch), normalize=True, nrow=40)
        torch.save(μ, '{}/particles_{}.pt'.format(args.modelp, epoch))
        fig, ax = plt.subplots()
        ax.set_ylabel('IPM estimate')
        ax.set_xlabel('iteration')
        ax.semilogy(losses)
        fig.savefig('{}/loss.png'.format(args.outf))
        plt.close(fig)
