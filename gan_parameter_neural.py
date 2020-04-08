import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.adamw import AdamW
import torch.utils.data
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from geomloss import SamplesLoss
from utils import base_module
from utils.sinkhorn_util import sinkhorn_potential
import time
import copy



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='celebA', help='dataset to be used')
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--n_particles', type=int, default=5000)
    parser.add_argument('--n_particles_neural', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=4001)
    parser.add_argument('--niterD', type=int, default=1, help='no. updates of D per update of G')
    parser.add_argument('--niterG', type=int, default=20, help='no. updates of G per update of D')
    parser.add_argument('--lr_encoder', type=float, default=2e-2, help='learning rate of c_encoder')
    parser.add_argument('--lr_decoder', type=float, default=2e-2, help='learning rate of μ_decoder')
    parser.add_argument('--alpha', type=float, default=0.0, help='Lagrange multiplier')
    parser.add_argument('--rho', type=float, default=1e-5, help='quadratic weight penalty')
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

    IMAGE_SIZE = 32
    η_g = 1 * (.9 ** -2)
    p = 2
    blur = 5
    λ = 0.1
    scaling = .95
    d_feature = 64
    kernel_parameter = 1

    time_start_0 = time.time()

    dataset = torch.load('{}/{}_transform_{}.pt'.format(args.dataf, args.dataset, IMAGE_SIZE))
    N_img, n_channels, IMAGE_SIZE_1, IMAGE_SIZE_2 = dataset.shape
    assert IMAGE_SIZE_1 == IMAGE_SIZE
    assert IMAGE_SIZE_2 == IMAGE_SIZE
    N_loop = int(N_img / args.batch_size)

    # initialize the decoder
    μ_decoder = base_module.Decoder(IMAGE_SIZE, n_channels, k=d_feature).to(device)
    μ_decoder.apply(base_module.weights_init)
    z = torch.FloatTensor(args.n_particles, d_feature, 1, 1).to(device)


    # all particles have uniform weights, pre-allocate the weight to cut cost
    μ_weight = torch.ones(args.n_particles, requires_grad=False).type_as(z) / args.n_particles
    x_real_weight = torch.ones(args.batch_size, requires_grad=False).type_as(z) / args.batch_size

    z_neural = torch.FloatTensor(args.n_particles_neural, d_feature, 1, 1).to(device)
    μ_weight_neural = torch.ones(args.n_particles_neural, requires_grad=False).type_as(z) / args.n_particles
    # initialize the encoder
    c_encoder = base_module.Encoder(IMAGE_SIZE, n_channels, k=d_feature).cuda()
    c_encoder.apply(base_module.weights_init)
    # c_encoder.load_state_dict(torch.load('{}/c_encoder_{}.pt'.format(args.modelf, 202)))

    optimizerC = optim.Adam(c_encoder.parameters(), lr=args.lr_encoder, betas=(0.5, 0.9), amsgrad=True)
    optimizerμ = optim.Adam(μ_decoder.parameters(), lr=args.lr_decoder, betas=(0.5, 0.9), amsgrad=True)
    # optimizerμ = AdamW(μ_decoder.parameters(), lr=args.lr_decoder, betas=(0.5, 0.9), amsgrad=True)


    # TODO: resolve the performance loss due to the call to max_diameter
    losses = []
    sinkhorn_divergence = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend="online", scaling=scaling)
    potential_operator = SamplesLoss(loss="sinkhorn", p=p, blur=blur, potentials=True, debias=False, backend="online",
                                     scaling=scaling)
    x_real_previous = None
    for epoch in range(args.epochs):
        time_start = time.time()
        loss = 0

        for i in range(N_loop):
            μ = μ_decoder(z.normal_(0, 1)).detach()
            x_real = dataset[torch.tensor(list(range(i * args.batch_size, (i + 1) * args.batch_size)))].to(device)
            if x_real_previous is None:
                x_real_previous = x_real

            # --- train the encoder of ground cost:
            # ---   optimizing on the ground space endowed with cost \|φ(x) - φ(y)\|^2 is equivalent to
            # ---   optimizing on the feature space with cost \|x - y\|^2.
            for iterD in range(args.niterD):
                # --- clip the weights of the encoder as regularization, otherwise the loss would blow
                for param in c_encoder.parameters():
                    param.data.clamp_(-0.01, 0.01)
                optimizerC.zero_grad()
                φ_x_real = c_encoder(x_real).view(-1, d_feature)
                φ_μ = c_encoder(μ).view(-1, d_feature)
                φ_x_real_previous = c_encoder(x_real_previous).view(-1, d_feature)
                negative_loss = -sinkhorn_divergence(x_real_weight, φ_x_real, μ_weight, φ_μ)\
                        + λ * sinkhorn_divergence(x_real_weight, φ_x_real, x_real_weight, φ_x_real_previous)
                negative_loss.backward()
                optimizerC.step()

            with torch.autograd.no_grad():
                φ_x_real = c_encoder(x_real).view(-1, d_feature)

            μ_particle = μ_decoder(z.normal_(0, 1))
            # μ.requries_grad = True
            φ_μ_particle = c_encoder(μ_particle).view(-1, d_feature)
            f_αβ, g_αβ = potential_operator(μ_weight, φ_μ_particle, x_real_weight, φ_x_real)
            f_αα, _ = potential_operator(μ_weight, φ_μ_particle, μ_weight, φ_μ_particle)
            g_ββ, _ = potential_operator(x_real_weight, φ_x_real_previous, x_real_weight, φ_x_real_previous)
            f_αβ_f_αα = f_αβ - f_αα
            g_αβ_g_ββ = g_αβ - g_ββ

            # f_αβ_2, _ = sinkhorn_potential(μ_weight, φ_μ, x_real_weight, φ_x_real, f_αβ, g_αβ, blur, p)


            # f_αβ_f_αα_gradient = torch.autograd.grad(torch.sum(f_αβ_f_αα), μ)[0]
            # μ = μ - η_g * f_αβ_f_αα_gradient
            # μ = μ.detach()

            with torch.autograd.no_grad():
                loss += torch.sum(f_αβ_f_αα).item() / args.n_particles + torch.sum(
                    g_αβ_g_ββ).item() / args.batch_size
            del f_αβ_f_αα, g_αβ_g_ββ

            loss_fn = torch.nn.MSELoss()
            μ_decoder_copy = copy.deepcopy(μ_decoder)
            for iterG in range(args.niterG):
                # for p in μ_decoder.parameters():
                #     p.data.clamp_(-0.5, 0.5)
                optimizerμ.zero_grad()
                μ = μ_decoder_copy(z_neural.normal_(0, 1))
                φ_μ = c_encoder(μ).view(-1, d_feature)
                f_αβ_neural = sinkhorn_potential(μ_weight_neural, φ_μ, x_real_weight, φ_x_real, f_αβ, g_αβ, blur, p)
                f_αα_neural = sinkhorn_potential(μ_weight_neural, φ_μ, μ_weight, φ_μ_particle, f_αα, f_αα, blur, p)
                f_αβ_f_αα_neural = f_αβ_neural - f_αα_neural
                f_αβ_f_αα_neural_gradient = torch.autograd.grad(torch.sum(f_αβ_f_αα_neural), μ)[0]
                μ = μ - η_g * f_αβ_f_αα_neural_gradient
                loss_fn(μ.detach(), μ_decoder(z_neural)).backward()
                optimizerμ.step()

            x_real_previous = x_real

        # shuffle the data after every epoch
        rand_perm_index = torch.randperm(N_img)
        dataset = dataset[rand_perm_index]

        loss /= N_loop
        losses.append(loss)

        print("epoch {0}, μ loss {1:9.3e}, time {2:9.3e}, total time {3:9.3e}".format(epoch, loss, time.time()-time_start,
              time.time() - time_start_0))
        # generated images and loss curve
        if epoch % 10 == 0:
            vutils.save_image(μ, '{}/x_{}.png'.format(args.outf, epoch), normalize=True, nrow=40)
            torch.save(c_encoder.state_dict(), '{}/c_encoder_{}.pt'.format(args.modelf, epoch))
            torch.save(μ_decoder.state_dict(), '{}/μ_decoder_{}.pt'.format(args.modelf, epoch))
        fig, ax = plt.subplots()
        ax.set_ylabel('IPM estimate')
        ax.set_xlabel('iteration')
        ax.semilogy(losses)
        fig.savefig('{}/loss.png'.format(args.outf))
        plt.close(fig)

        # assert(epoch < 10)
