import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from geomloss import SamplesLoss
from utils import base_module_high_dim_encoder
import time
from torchsummary import summary
from tqdm import tqdm
from pykeops.torch import LazyTensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='celebA', help='dataset to be used')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--n_particles', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=4001)
    parser.add_argument('--niterD', type=int, default=1, help='no. updates of D per update of G')
    parser.add_argument('--niterG', type=int, default=3, help='no. updates of G per update of D')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--backend', default='tensorized', help='backend of geomloss')
    args = parser.parse_args()

    cudnn.benchmark = True

    args.outf = "tmp/{}".format(args.dataset)
    args.modelf = "model/{}".format(args.dataset)
    args.particlef = "particle/{}".format(args.dataset)
    args.dataf = "data_transform/{}".format(args.dataset)

    os.system('mkdir -p {}'.format(args.outf))
    os.system('mkdir -p {}'.format(args.modelf))
    os.system('mkdir -p {}'.format(args.particlef))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_SIZE = args.image_size
    η_g = 2
    p = 2
    blur = 10
    λ = 0
    scaling = .95
    kernel_parameter = 1
    backend = args.backend
    time_start_0 = time.time()

    dataset = torch.load('{}/{}_transform_{}.pt'.format(args.dataf, args.dataset, IMAGE_SIZE))
    N_img, n_channels, IMAGE_SIZE_1, IMAGE_SIZE_2 = dataset.shape
    assert IMAGE_SIZE_1 == IMAGE_SIZE
    assert IMAGE_SIZE_2 == IMAGE_SIZE
    N_loop = int(N_img / args.batch_size)

    # random particles as the initial measure
    μ = torch.cuda.FloatTensor(args.n_particles, n_channels, IMAGE_SIZE_1, IMAGE_SIZE_2).uniform_(-1, 1)
    μ.requires_grad_()

    # all particles have uniform weights, pre-allocate the weight to cut cost
    μ_weight = torch.ones(args.n_particles, requires_grad=False).type_as(μ) / args.n_particles
    x_real_weight = torch.ones(args.batch_size, requires_grad=False).type_as(μ) / args.batch_size

    # initialize the encoder
    c_encoder = base_module_high_dim_encoder.Encoder(IMAGE_SIZE, n_channels).cuda()
    c_encoder.apply(base_module_high_dim_encoder.weights_init)

    optimizerC = optim.Adam(c_encoder.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=False)


    summary(c_encoder, (n_channels, IMAGE_SIZE, IMAGE_SIZE))
    losses = []
    sinkhorn_divergence = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend=backend, scaling=scaling)
    potential_operator = SamplesLoss(loss="sinkhorn", p=p, blur=blur, potentials=True, debias=True, backend=backend,
                                     scaling=scaling)
    x_real_previous = None
    for epoch in range(args.epochs):
        time_start = time.time()
        loss = 0
        G_count = 0
        for i in tqdm(range(N_loop)):
            # --- avoid unnecessary backward
            data = dataset[torch.tensor(list(range(i * args.batch_size, (i + 1) * args.batch_size)))]
            x_real = data.to(device)
            if i % (args.niterG+1) is 0:
                μ_detach = μ.detach()
                # --- train the encoder of ground cost:
                # ---   optimizing on the ground space endowed with cost \|φ(x) - φ(y)\|^2 is equivalent to
                # ---   optimizing on the feature space with cost \|x - y\|^2.
                for iterD in range(args.niterD):
                    optimizerC.zero_grad()
                    φ_x_real = c_encoder(x_real)
                    φ_μ = c_encoder(μ_detach)
                    negative_loss = -sinkhorn_divergence(x_real_weight, φ_x_real, μ_weight, φ_μ)
                    negative_loss.backward()
                    optimizerC.step()
            else:
                G_count += 1
                with torch.autograd.no_grad():
                    φ_x_real = c_encoder(x_real)

                # --- train particles of μ
                φ_μ = c_encoder(μ)
                f_αβ_f_αα, g_αβ_g_αα = potential_operator(μ_weight, φ_μ, x_real_weight, φ_x_real)
                f_αβ_f_αα_gradient = torch.autograd.grad(torch.sum(f_αβ_f_αα), μ)[0]

                μ = μ - η_g * f_αβ_f_αα_gradient

                with torch.autograd.no_grad():
                    loss += torch.sum(f_αβ_f_αα).item() / args.n_particles + torch.sum(
                        g_αβ_g_αα).item() / args.batch_size

                del φ_μ, f_αβ_f_αα, f_αβ_f_αα_gradient, g_αβ_g_αα

        # shuffle the data after every epoch
        rand_perm_index = torch.randperm(N_img)
        dataset = dataset[rand_perm_index]

        loss = loss / G_count
        losses.append(loss)

        print("epoch {0}, μ loss {1:9.3e}, time {2:9.3e}, total time {3:9.3e}".format(epoch, loss, time.time()-time_start,
              time.time() - time_start_0))
        print("min {0:9.3e} and max {1:9.3e}".format(torch.min(μ), torch.max(μ)))
        # generated images and loss curve
        vutils.save_image(μ, '{}/x_{}.png'.format(args.outf, epoch), normalize=True, nrow=40)
        torch.save(c_encoder.state_dict(), '{}/c_encoder_{}.pt'.format(args.modelf, epoch))
        torch.save(μ, '{}/particles_{}.pt'.format(args.particlef, epoch))
        fig, ax = plt.subplots()
        ax.set_ylabel('IPM estimate')
        ax.set_xlabel('iteration')
        ax.semilogy(losses)
        fig.savefig('{}/loss.png'.format(args.outf))
        plt.close(fig)
