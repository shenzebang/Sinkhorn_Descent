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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outf', default='tmp/mnist', help='where to save results')
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--n_particles', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--niterD', type=int, default=10, help='no. updates of D per update of G')
    parser.add_argument('--niterG', type=int, default=2, help='no. updates of G per update of D')
    parser.add_argument('--niterG_final', type=int, default=200, help='no. updates of G in the end')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.0, help='Lagrange multiplier')
    parser.add_argument('--rho', type=float, default=1e-5, help='quadratic weight penalty')
    args = parser.parse_args()

    cudnn.benchmark = True

    os.system('mkdir -p {}'.format(args.outf))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_SIZE = 32
    n_channels = 1
    η = 1
    p = 2
    blur = 5
    scaling = .95
    d = n_channels*IMAGE_SIZE*IMAGE_SIZE
    d_feature = 16
    noise_level = 0.1
    c_shape = [-1, n_channels, IMAGE_SIZE, IMAGE_SIZE]
    dataset = dset.MNIST(root='mnist', download=True, train=True,
                         transform=transforms.Compose([
                             transforms.Resize(IMAGE_SIZE),
                             transforms.ToTensor(),
                             # first, convert image to PyTorch tensor
                             transforms.Normalize((0.1307,), (0.3081,))
                             # normalize inputs
                         ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=2, drop_last=True)

    # random particles as the initial measure
    μ = torch.rand(args.n_particles, n_channels, IMAGE_SIZE, IMAGE_SIZE).cuda()
    μ = μ * 2 - 1
    μ.requires_grad_()
    # normalize the random particles to [-1, 1]
    # μ = μ*2-1

    c_encoder = base_module.Encoder(IMAGE_SIZE, n_channels, k=d_feature).cuda()
    optimizerC = optim.Adam(c_encoder.parameters(), lr=args.lr, betas=(0.5, 0.9), amsgrad=True)
    optimizerμ = optim.SGD([μ], lr=1)
    # optimizerμ = optim.Adam([μ], lr=1, betas=(0.5, 0.9), amsgrad=True)

    losses = []
    sinkhorn_divergence = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend="online", scaling=scaling)
    potential_operator = SamplesLoss(loss="sinkhorn", p=p, blur=blur, potentials=True, debias=True, backend="online", scaling=scaling)
    for epoch in range(args.epochs):
        print("epoch {}".format(epoch))
        for i, data in enumerate(dataloader):
            # --- avoid unnecessary backward
            μ_detach = μ.detach()

            # --- move the data to the device (CUDA)
            x_real = data[0].to(device)

            # --- train the encoder of ground cost:
            # ---   optimizing on the ground space endowed with cost \|φ(x) - φ(y)\|^2 is equivalent to
            # ---   optimizing on the feature space with cost \|x - y\|^2.
            for iterD in range(args.niterD):
                # --- clip the weights of the encoder as regularization, otherwise the loss would blow
                for p in c_encoder.parameters():
                    p.data.clamp_(-0.01, 0.01)
                optimizerC.zero_grad()
                φ_x_real = c_encoder(x_real).view(-1, d_feature)
                φ_μ = c_encoder(μ_detach).view(-1, d_feature)
                negative_loss = -sinkhorn_divergence(φ_x_real, φ_μ)
                negative_loss.backward()
                optimizerC.step()

            with torch.autograd.no_grad():
                φ_x_real = c_encoder(x_real).view(-1, d_feature)

            for iterG in range(args.niterG):
                # --- train particles of μ
                φ_μ = c_encoder(μ).view(-1, d_feature)
                with torch.autograd.no_grad():
                    loss_before_sd = sinkhorn_divergence(φ_x_real, φ_μ)

                f_αβ_f_αα, _ = potential_operator(φ_μ, φ_x_real)
                torch.sum(f_αβ_f_αα).backward()
                optimizerμ.step()
                losses.append(loss_before_sd.item())
                print("\t data loop {}, before update μ loss {}".format(i, loss_before_sd))
                del φ_μ, f_αβ_f_αα

        # generated images and loss curve
        vutils.save_image(μ, '{}/x_{}.png'.format(args.outf, epoch), normalize=True, nrow=40)
        fig, ax = plt.subplots()
        ax.set_ylabel('IPM estimate')
        ax.set_xlabel('iteration')
        ax.semilogy(losses)
        fig.savefig('{}/loss.png'.format(args.outf))
        plt.close(fig)

        noise_level *= 0.9



