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
from geomloss.sinkhorn_divergence import max_diameter
from utils import base_module
import time
from pykeops.torch import LazyTensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outf', default='tmp/celebrity', help='where to save results')
    parser.add_argument('--modelf', default='model/celebrity', help='where to save cost model')
    parser.add_argument('--particlef', default='particle/celebrity', help='where to save particles')
    parser.add_argument('--dataf', default='data_transform/celebrity', help='where to save transformed data')
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--n_particles', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--niterD', type=int, default=1, help='no. updates of D per update of G')
    parser.add_argument('--niterG', type=int, default=1, help='no. updates of G per update of D')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.0, help='Lagrange multiplier')
    parser.add_argument('--rho', type=float, default=1e-5, help='quadratic weight penalty')
    args = parser.parse_args()

    cudnn.benchmark = True

    os.system('mkdir -p {}'.format(args.outf))
    os.system('mkdir -p {}'.format(args.modelf))
    os.system('mkdir -p {}'.format(args.particlef))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_SIZE = 32
    η_g = 1
    p = 2
    blur = 5
    scaling = .95
    d_feature = 32
    kernel_parameter = 1

    time_start_0 = time.time()


    dataset = torch.load('{}/celebA_transform_{}.pt'.format(args.dataf, IMAGE_SIZE))
    N_img, n_channels, IMAGE_SIZE_1, IMAGE_SIZE_2 = dataset.shape
    N_loop = int(N_img / args.batch_size)

    # initialize the decoder
    μ_decoder = base_module.Decoder(IMAGE_SIZE, n_channels, k=d_feature).to(device)
    z = torch.FloatTensor(args.n_particles, d_feature, 1, 1).to(device)


    # all particles have uniform weights, pre-allocate the weight to cut cost
    μ_weight = torch.ones(args.n_particles, requires_grad=False).type_as(z) / args.n_particles
    x_real_weight = torch.ones(args.batch_size, requires_grad=False).type_as(z) / args.batch_size

    # initialize the encoder
    c_encoder = base_module.Encoder(IMAGE_SIZE, n_channels, k=d_feature).cuda()

    # c_encoder.load_state_dict(torch.load('{}/c_encoder_{}.pt'.format(args.modelf, 202)))

    # initialize the optimizers
    optimizerC = optim.Adam(c_encoder.parameters(), lr=args.lr, betas=(0.5, 0.9), amsgrad=True)
    optimizerμ = optim.Adam(μ_decoder.parameters(), lr=args.lr, betas=(0.5, 0.9), amsgrad=True)

    # losses = []
    # TODO: resolve the performance loss due to the call to max_diameter
    losses = []
    sinkhorn_divergence = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend="online", scaling=scaling)
    potential_operator = SamplesLoss(loss="sinkhorn", p=p, blur=blur, potentials=True, debias=True, backend="online",
                                     scaling=scaling)
    for epoch in range(args.epochs):
        # print("epoch {}".format(epoch))
        time_start = time.time()
        # for i, data in enumerate(dataloader):
        loss = 0

        for i in range(N_loop):

            # --- avoid unnecessary backward
            μ_detach = μ_decoder(z.normal_(0, 1)).detach()
            data = dataset[torch.tensor(list(range(i * args.batch_size, (i + 1) * args.batch_size)))]
            # --- move the data to the device (CUDA)
            x_real = data.to(device)

            # with torch.autograd.no_grad():
            #     if φ_1 is None:
            #         φ_1 = c_encoder(x_real).squeeze()
            #     φ_2 = c_encoder(x_real).squeeze()
            #     # print(φ_1.shape)
            #     d_real = sinkhorn_divergence(x_real_weight, φ_1, x_real_weight, φ_2)
            #     φ_1 = φ_2
            #     del φ_2


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
                negative_loss = -sinkhorn_divergence(x_real_weight, φ_x_real, μ_weight, φ_μ)
                # del φ_x_real, φ_μ
                negative_loss.backward()
                optimizerC.step()
                # print("\t \t cost loop {}, loss {}".format(iterD, -negative_loss))

            with torch.autograd.no_grad():
                φ_x_real = c_encoder(x_real).view(-1, d_feature)

            # print(max_diameter(φ_x_real, φ_x_real))

            for iterG in range(args.niterG):
                # --- train particles of μ
                # with torch.autograd.no_grad():
                optimizerμ.zero_grad()
                μ = μ_decoder(z.normal_(0, 1))
                φ_μ = c_encoder(μ).view(-1, d_feature)
                # with torch.autograd.no_grad():
                #     loss_before_sd = sinkhorn_divergence(x_real_weight, φ_x_real, μ_weight, φ_μ)

                f_αβ_f_αα, g_αβ_g_αα = potential_operator(μ_weight, φ_μ, x_real_weight, φ_x_real)
                torch.sum(f_αβ_f_αα).backward()

                # gradient = f_αβ_f_αα_gradient + noise_level * torch.randn_like(f_αβ_f_αα_gradient)
                # gradient = f_αβ_f_αα_gradient
                # # convolve with the gaussian kernel
                # μ_detach = μ.detach().view([-1, d])/kernel_parameter
                # x_i = LazyTensor(μ_detach[:, None, :])  # (M, 1, D) LazyTensor
                # y_j = LazyTensor(μ_detach[None, :, :])
                #
                # D_ij = ((x_i - y_j) ** 2).sum(-1)  # Symbolic (N, N) matrix of square distances
                # K_ij = (- D_ij/2).exp()  # Symbolic (M, N) Laplacian (aka. exponential) kernel matrix
                # gradient = K_ij @ f_αβ_f_αα_gradient.view([-1, d])
                # gradient = gradient.view(c_shape)

                # μ = μ - η_g * (.8 ** -2) * f_αβ_f_αα_gradient
                # μ_old = μ.clone()
                # for ls in range(-2, 10):
                #     μ = μ - η_g * (.8 ** ls) * gradient
                #     with torch.autograd.no_grad():
                #         φ_μ = c_encoder(μ).view(-1, d_feature)
                #         loss_after_sd = sinkhorn_divergence(x_real_weight, φ_x_real, μ_weight, φ_μ)
                #         if loss_after_sd > loss_before_sd:
                #             μ = μ_old.clone()
                #         else:
                #             # print(ls)
                #             break
                # loss += loss_after_sd.item()

                # print("epoch {0}, loop {1}, μ loss {2:9.3e}, ls {3}, d real {4:9.3e}".format(epoch, i, loss_before_sd, ls, d_real))
                # print("epoch {0}, loop {1}, μ loss {2:9.3e}, ls {3}".format(epoch, i, loss_before_sd, ls))
                # μ += torch.randn_like(μ) * noise_level
                # --- logging
                # del φ_μ, f_αβ_f_αα, f_αβ_f_αα_gradient, loss_after_sd, \
                #     loss_before_sd, μ_old, D_ij, K_ij, μ_detach, x_i, y_j, gradient
                # del φ_μ, f_αβ_f_αα, f_αβ_f_αα_gradient, loss_after_sd, loss_before_sd, μ_old

                with torch.autograd.no_grad():
                    loss += torch.sum(f_αβ_f_αα).item() / args.n_particles + torch.sum(
                        g_αβ_g_αα).item() / args.batch_size

                del φ_μ, f_αβ_f_αα, g_αβ_g_αα


        # shuffle the data after every epoch
        rand_perm_index = torch.randperm(N_img)
        dataset = dataset[rand_perm_index]

        loss /= N_loop
        losses.append(loss)

        print("epoch {0}, μ loss {1:9.3e}, time {2:9.3e}, total time {3:9.3e}".format(epoch, loss, time.time()-time_start,
              time.time() - time_start_0))
        # generated images and loss curve
        if epoch % 10 == 0:
            μ_detach = μ_decoder(z.normal_(0, 1)).detach()
            vutils.save_image(μ_detach, '{}/x_{}.png'.format(args.outf, epoch), normalize=True, nrow=40)
        torch.save(c_encoder.state_dict(), '{}/c_encoder_{}.pt'.format(args.modelf, epoch))
        torch.save(μ_decoder.state_dict(), '{}/μ_decoder_{}.pt'.format(args.particlef, epoch))
        fig, ax = plt.subplots()
        ax.set_ylabel('IPM estimate')
        ax.set_xlabel('iteration')
        ax.semilogy(losses)
        fig.savefig('{}/loss.png'.format(args.outf))
        plt.close(fig)

        # assert(epoch < 10)
