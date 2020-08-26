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
from utils import base_module_ot_gan
from utils.sinkhorn_util import sinkhorn_potential, potential_operator_grad
from torchsummary import summary
import time
import math
from tqdm import tqdm
import torch.utils.data as Data
from utils.conjugate_gradient import conjugate_gradients, set_flat_params_to, get_flat_params_from
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    problem_name = 'distribution_approximation'
    algorithm_name = 'SiNG_ADAM'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='celebA', help='dataset to be used')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--n_particles', type=int, default=4000)
    parser.add_argument('--n_observe', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=4001)
    parser.add_argument('--niterG', type=int, default=2, help='no. updates of G per update of D')
    parser.add_argument('--decoder_z_features', type=int, default=64)
    parser.add_argument('--eta', type=float, default=30, help='step size of the natural gradient update')
    parser.add_argument('--damping', type=float, default=.05, help='damping parameter of natural gradient')
    parser.add_argument('--max_sd', type=float, default=.1, help='maximum allowed sd parameter of natural gradient')
    parser.add_argument('--scaling', type=float, default=.95, help='scaling parameter for the Geomloss package')
    parser.add_argument('--generator_backend', default='DC-GAN', help='NN model of the generator')
    parser.add_argument('--lr_decoder', type=float, default=5e-3, help='learning rate of μ_decoder')

    args = parser.parse_args()

    cudnn.benchmark = True
    args.outf = "output/{}/{}/{}".format(problem_name, algorithm_name, args.dataset)
    args.modelf = "model/{}/{}/{}".format(problem_name, algorithm_name, args.dataset)
    args.dataf = "data_transform"

    if not os.path.exists(args.outf): os.makedirs(args.outf)
    if not os.path.exists(args.modelf): os.makedirs(args.modelf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(args.outf)
    IMAGE_SIZE = args.image_size
    η_g = args.eta
    p = 2
    blur_loss = .1
    scaling = args.scaling
    decoder_z_features = args.decoder_z_features
    backend = "tensorized"

    time_start_0 = time.time()
    dataset = torch.load('{}/{}.pt'.format(args.dataf, args.dataset))
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
    z_observe = torch.cuda.FloatTensor(args.n_observe, decoder_z_features, 1, 1).normal_(0, 1)
    weight = torch.ones(args.n_particles, requires_grad=False).type_as(z) / args.n_particles
    weight_half = torch.ones(args.n_particles // 2, requires_grad=False).type_as(z) / (args.n_particles / 2)
    optimizerμ = optim.Adam(μ_decoder.parameters(), lr=args.lr_decoder, betas=(0.5, 0.999), amsgrad=False)

    summary(μ_decoder, (decoder_z_features, 1, 1))

    # losses = []

    sinkhorn_divergence = SamplesLoss(loss="sinkhorn", p=p, blur=blur_loss, backend=backend, scaling=scaling, debias=True)
    total_steps = 0
    time_Adam = 0
    for epoch in range(args.epochs):
        i = 0
        for data in tqdm(loader):
            time_start = time.time()
            optimizerμ.zero_grad()
            x_real = data[0].to(device).view(args.batch_size, -1)
            μ = μ_decoder(z.normal_(0, 1)).view(args.batch_size, -1)
            loss_G = sinkhorn_divergence(weight, μ, weight, x_real)
            loss_print = loss_G.item()
            loss_G.backward()
            optimizerμ.step()
            time_Adam += time.time() - time_start

            i += 1
            total_steps += 1




            # print("epoch {0}, μ loss {1:9.3e}, time {2:9.3e}, total time {3:9.3e}".format(epoch, loss_print, time.time()-time_start,
            #       time.time() - time_start_0))
            writer.add_scalar('loss_blur_{} lr_{} sg/loss'.format(blur_loss, args.lr_decoder), loss_print, total_steps)
            writer.add_scalar('loss_blur_{} lr_{} sg/time'.format(blur_loss, args.lr_decoder), time_Adam, total_steps)
            writer.flush()
            # generated images and loss curve
            nrow = int(math.sqrt(args.n_observe))
            vutils.save_image(μ_decoder(z_observe), '{}/x_{}_{}.png'.format(args.outf, epoch, i), normalize=True, nrow=nrow)
            # fig, ax = plt.subplots()
            # ax.set_ylabel('IPM estimate')
            # ax.set_xlabel('iteration')
            # ax.semilogy(losses)
            # fig.savefig('{}/loss.png'.format(args.outf))
            # plt.close(fig)
        torch.save(μ_decoder.state_dict(), '{}/μ_decoder_{}.pt'.format(args.modelf, epoch))
