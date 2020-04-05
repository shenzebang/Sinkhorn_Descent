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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outf', default='tmp/cifar10', help='where to save results')
    parser.add_argument('--dataf', default='data_transform/cifar10', help='where the transformed data is saved')
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--n_particles', type=int, default=1600)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--niterD', type=int, default=1, help='no. updates of D per update of G')
    parser.add_argument('--niterG', type=int, default=1, help='no. updates of G per update of D')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.0, help='Lagrange multiplier')
    parser.add_argument('--rho', type=float, default=1e-5, help='quadratic weight penalty')
    args = parser.parse_args()

    cudnn.benchmark = True

    os.system('mkdir -p {}'.format(args.outf))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_SIZE = 32
    n_channels = 3
    η = 1
    p = 2
    blur = 5
    scaling = .95
    noise_level = 0.1
    d = n_channels*IMAGE_SIZE*IMAGE_SIZE
    d_feature = 64
    c_shape = [-1, n_channels, IMAGE_SIZE, IMAGE_SIZE]
    # dataset = dset.CIFAR10(root='cifar10', download=True,
    #                        transform=transforms.Compose([
    #                            transforms.Resize(IMAGE_SIZE),
    #                            transforms.ToTensor(),
    #                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                        ]))
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
    #                                          shuffle=True, num_workers=8, drop_last=True)

    dataset = torch.load('{}/cifar10_transform.pt'.format(args.dataf))
    N_img, n_channels_load, IMAGE_SIZE_1, IMAGE_SIZE_2 = dataset.shape
    assert n_channels == n_channels_load
    assert IMAGE_SIZE_1 == IMAGE_SIZE
    assert IMAGE_SIZE_2 == IMAGE_SIZE
    N_loop = int(N_img / args.batch_size)
    # random particles as the initial measure
    μ = torch.rand(args.n_particles, n_channels, IMAGE_SIZE, IMAGE_SIZE).cuda()
    μ = μ * 2 - 1
    μ.requires_grad_()

    c_encoder = base_module.Encoder(IMAGE_SIZE, n_channels, k=d_feature).cuda()
    optimizerC = optim.Adam(c_encoder.parameters(), lr=args.lr, betas=(0.5, 0.9), amsgrad=True)

    losses = []
    sinkhorn_divergence = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend="online", scaling=scaling)
    potential_operator = SamplesLoss(loss="sinkhorn", p=p, blur=blur, potentials=True, debias=True, backend="online",
                                     scaling=scaling)
    for epoch in range(args.epochs):
        time_start = time.time()
        loss = 0

        for i in range(N_loop):
            # --- avoid unnecessary backward
            μ_detach = μ.detach()
            data = dataset[torch.tensor(list(range(i*args.batch_size, (i+1)*args.batch_size)))]
            # print(data.shape)
            # assert 0 == 1
            # --- move the data to the device (CUDA)
            x_real = data.to(device)

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
                # del φ_x_real, φ_μ
                negative_loss.backward()
                optimizerC.step()
                # print("\t \t cost loop {}, loss {}".format(iterD, -negative_loss))

            with torch.autograd.no_grad():
                φ_x_real = c_encoder(x_real).view(-1, d_feature)

            for iterG in range(args.niterG):
                # --- train particles of μ
                # with torch.autograd.no_grad():
                φ_μ = c_encoder(μ).view(-1, d_feature)
                with torch.autograd.no_grad():
                    loss_before_sd = sinkhorn_divergence(φ_x_real, φ_μ)

                # φ_x_real = c_encoder(x_real).view(-1, d_feature)
                # μ_noise = μ + torch.randn_like(μ, requires_grad=False) * noise_level
                # φ_μ_noise = c_encoder(μ_noise).view(-1, d_feature)
                f_αβ_f_αα, g_αβ_g_αα = potential_operator(φ_μ, φ_x_real)
                f_αβ_f_αα_gradient = torch.autograd.grad(torch.sum(f_αβ_f_αα), μ)[0]

                # φ_μ_noise = c_encoder(μ_noise).view(-1, d_feature)
                # f_αα, _ = potential_operator(φ_μ, φ_μ)
                # f_αα_gradient = torch.autograd.grad(torch.sum(f_αα), μ)[0]

                # μ_old = μ.clone()
                # for ls in range(-2, 10):
                μ = μ - η * (.8 ** -2) * f_αβ_f_αα_gradient
                    # with torch.autograd.no_grad():
                    #     φ_μ = c_encoder(μ).view(-1, d_feature)
                    #     loss_after_sd = sinkhorn_divergence(φ_x_real, φ_μ)
                    #     if loss_after_sd > loss_before_sd:
                    #         μ = μ_old.clone()
                    #     else:
                    #         break
                # print("\t data loop {}, after update μ loss {}".format(i, loss_before_sd))
                # μ += torch.randn_like(μ) * noise_level
                # --- logging
                # loss += loss_after_sd.item()
                with torch.autograd.no_grad():
                    loss += torch.sum(f_αβ_f_αα).item() / args.n_particles + torch.sum(
                        g_αβ_g_αα).item() / args.batch_size

                del φ_μ, f_αβ_f_αα, f_αβ_f_αα_gradient, loss_before_sd

        # shuffle the data after every epoch
        rand_perm_index = torch.randperm(N_img)
        dataset = dataset[rand_perm_index]

        # record and report loss
        loss /= N_loop
        losses.append(loss)
        print("epoch {0}, μ loss {1:9.3e}, time {2}".format(epoch, loss, time.time()-time_start))

        # generated images and loss curve
        vutils.save_image(μ, '{}/x_{}.png'.format(args.outf, epoch), normalize=True, nrow=40)
        fig, ax = plt.subplots()
        ax.set_ylabel('IPM estimate')
        ax.set_xlabel('iteration')
        ax.semilogy(losses)
        fig.savefig('{}/loss.png'.format(args.outf))
        plt.close(fig)
