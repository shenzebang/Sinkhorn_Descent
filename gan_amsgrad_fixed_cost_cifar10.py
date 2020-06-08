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
from utils import base_module_high_dim_encoder
from utils import base_module_ot_gan
from utils.sinkhorn_util import sinkhorn_potential, potential_operator_grad
from torchsummary import summary
import time
import copy
import math
from tqdm import tqdm
import torch.utils.data as Data
from utils.conjugate_gradient import conjugate_gradients, set_flat_params_to, get_flat_params_from


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', help='dataset to be used')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=3000)
    parser.add_argument('--n_particles', type=int, default=3000)
    parser.add_argument('--n_particles_neural', type=int, default=100)
    parser.add_argument('--n_observe', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=4001)
    parser.add_argument('--niterD', type=int, default=1, help='no. updates of D per update of G')
    parser.add_argument('--niterG', type=int, default=2, help='no. updates of G per update of D')
    parser.add_argument('--decoder_z_features', type=int, default=64)
    parser.add_argument('--lr_encoder', type=float, default=1e-3, help='learning rate of c_encoder')
    parser.add_argument('--lr_decoder', type=float, default=1e-3, help='learning rate of μ_decoder')
    parser.add_argument('--eta', type=float, default=30, help='step size of the particle update')
    parser.add_argument('--damping', type=float, default=.05, help='damping parameter of natural gradient')
    parser.add_argument('--max_sd', type=float, default=.1, help='maximum allowed sd parameter of natural gradient')
    parser.add_argument('--scaling', type=float, default=.95, help='scaling parameter for the Geomloss package')
    parser.add_argument('--generator_backend', default='DC-GAN', help='NN model of the generator')
    parser.add_argument('--load_whole_dataset', action='store_true', help='This can deplete GPU memory easily')

    args = parser.parse_args()
    print(args)

    cudnn.benchmark = True
    args.outf = "tmp/{}".format(args.dataset)
    args.modelf = "model/{}".format(args.dataset)
    args.particlef = "particle/{}".format(args.dataset)
    args.dataf = "data_transform/{}".format(args.dataset)
    args.plotf = "plot/gan/{}".format(args.dataset)

    os.system('mkdir -p {}'.format(args.outf))
    os.system('mkdir -p {}'.format(args.modelf))
    os.system('mkdir -p {}'.format(args.particlef))
    os.system('mkdir -p {}'.format(args.plotf))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_SIZE = args.image_size
    η_g = args.eta
    p = 2
    blur = 10
    scaling = args.scaling
    decoder_z_features = args.decoder_z_features
    backend = "tensorized"

    time_start_0 = time.time()
    dataset = torch.load('{}/{}_transform_{}.pt'.format(args.dataf, args.dataset, IMAGE_SIZE))
    N_img, n_channels, IMAGE_SIZE_1, IMAGE_SIZE_2 = dataset.shape
    assert IMAGE_SIZE_1 == IMAGE_SIZE
    assert IMAGE_SIZE_2 == IMAGE_SIZE
    N_loop = int(N_img / args.batch_size)
    # Load dataset as TensorDataset
    dataset = Data.TensorDataset(dataset)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )

    # initialize the decoder
    if args.generator_backend is 'DC-GAN':
        μ_decoder = base_module_high_dim_encoder.Decoder(IMAGE_SIZE, n_channels, k=decoder_z_features).to(device)
    elif args.generator_backend is 'OT-GAN':
        μ_decoder = base_module_ot_gan.Decoder(IMAGE_SIZE, n_channels, k=decoder_z_features).to(device).to(device)
    else:
        print("generator unknown")
        μ_decoder = None
    μ_decoder.apply(base_module_ot_gan.weights_init)
    z = torch.cuda.FloatTensor(args.n_particles, decoder_z_features, 1, 1)
    z_neural = torch.FloatTensor(args.n_particles_neural, decoder_z_features, 1, 1).to(device)
    z_observe = torch.cuda.FloatTensor(args.n_observe, decoder_z_features, 1, 1).normal_(0, 1)
    μ_weight = torch.ones(args.n_particles, requires_grad=False).type_as(z) / args.n_particles

    x_real_weight = torch.ones(args.batch_size, requires_grad=False).type_as(z) / args.batch_size


    # initialize the encoder
    c_encoder = base_module_high_dim_encoder.Encoder(IMAGE_SIZE, n_channels).cuda()
    c_encoder.load_state_dict(torch.load('{}/c_encoder_{}.pt'.format(args.modelf, 100)))

    optimizerμ = optim.Adam(μ_decoder.parameters(), lr=args.lr_decoder, betas=(0.5, 0.999), amsgrad=True)


    # summary(c_encoder, (n_channels, IMAGE_SIZE, IMAGE_SIZE))
    summary(μ_decoder, (decoder_z_features, 1, 1))

    losses = []
    loss_G_list = []
    n_epoch_list = []
    n_iteration = 0
    sinkhorn_divergence = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend=backend, scaling=scaling)
    potential_operator = SamplesLoss(loss="sinkhorn", p=p, blur=blur, potentials=True, debias=False, backend=backend,
                                     scaling=scaling)
    for epoch in range(args.epochs):
        time_start = time.time()
        loss = 0
        G_count = 0
        for data in tqdm(loader):
            x_real = data[0].to(device)
            G_count += 1
            # train the decoder with natural gradient
            # 0. construct the discrete approximate by sampling
            with torch.autograd.no_grad():
                φ_x_real = c_encoder(x_real)
            φ_μ = c_encoder(μ_decoder(z.normal_(0, 1)))
            # 1. compute the gradient of the free energy
            optimizerμ.zero_grad()
            loss_sinkhorn = sinkhorn_divergence(μ_weight, φ_μ, x_real_weight, φ_x_real)
            loss_sinkhorn.backward()
            optimizerμ.step()

            with torch.autograd.no_grad():
                φ_μ_particle = c_encoder(μ_decoder(z))
                loss_G = sinkhorn_divergence(x_real_weight, φ_x_real, μ_weight, φ_μ_particle)
                loss += loss_G.item()
                loss_G_list.append(loss_G.item())
                n_iteration += 1
                n_epoch_list.append(n_iteration)
            print("epoch {0}, inner loop, μ loss {1:9.3e}".format(epoch, loss_G.item()))

        loss /= G_count
        losses.append(loss)

        print("epoch {0}, μ loss {1:9.3e}, time {2:9.3e}, total time {3:9.3e}".format(epoch, loss, time.time()-time_start,
              time.time() - time_start_0))
        loss_G_tensor = torch.FloatTensor(loss_G_list)
        n_epoch_list_tensor = torch.FloatTensor(n_epoch_list) / len(n_epoch_list) * (epoch+1)
        result = torch.stack([loss_G_tensor, n_epoch_list_tensor])
        torch.save(result, '{}/amsgrad.pt'.format(args.plotf))
        # generated images and loss curve
        # nrow = int(math.sqrt(args.n_observe))
        # vutils.save_image(μ_decoder(z_observe), '{}/x_{}.png'.format(args.outf, epoch), normalize=True, nrow=nrow)
        # # torch.save(c_encoder.state_dict(), '{}/c_encoder_{}.pt'.format(args.modelf, epoch))
        # # torch.save(μ_decoder.state_dict(), '{}/μ_decoder_{}.pt'.format(args.modelf, epoch))
        # fig, ax = plt.subplots()
        # ax.set_ylabel('IPM estimate')
        # ax.set_xlabel('iteration')
        # ax.semilogy(losses)
        # fig.savefig('{}/loss_eg.png'.format(args.outf))
        # plt.close(fig)
