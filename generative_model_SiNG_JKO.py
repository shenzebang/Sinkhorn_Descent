import argparse
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')

from geomloss import SamplesLoss
from utils import base_module_high_dim_encoder
from utils import base_module_ot_gan
from torchsummary import summary
import time
import math
from tqdm import tqdm
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    problem_name = 'distribution_approximation'
    algorithm_name = 'SiNG_JKO'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='celebA', help='dataset to be used')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--n_particles', type=int, default=4000)
    parser.add_argument('--n_observe', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=4001)
    parser.add_argument('--niterG', type=int, default=2, help='no. updates of G per update of D')
    parser.add_argument('--lr_decoder', type=float, default=1e-3, help='learning rate of μ_decoder')
    parser.add_argument('--decoder_z_features', type=int, default=64)
    parser.add_argument('--eta', type=float, default=1, help='step size of the natural gradient update')
    parser.add_argument('--damping', type=float, default=.05, help='damping parameter of natural gradient')
    parser.add_argument('--max_sd', type=float, default=.1, help='maximum allowed sd parameter of natural gradient')
    parser.add_argument('--scaling', type=float, default=.95, help='scaling parameter for the Geomloss package')
    parser.add_argument('--potential_iter', type=int, default=20, help='no. sinkhorn mapping to accumulate the potential')
    parser.add_argument('--generator_backend', default='DC-GAN', help='NN model of the generator')
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
    blur_constraint = .1
    jko_steps = 20
    scaling = args.scaling
    decoder_z_features = args.decoder_z_features
    backend = "tensorized"
    potential_iter = args.potential_iter

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
    optimizerμ = optim.Adam(μ_decoder.parameters(), lr=args.lr_decoder, betas=(0.5, 0.999), amsgrad=False)


    z = torch.cuda.FloatTensor(args.n_particles, decoder_z_features, 1, 1)
    z_observe = torch.cuda.FloatTensor(args.n_observe, decoder_z_features, 1, 1).normal_(0, 1)
    weight = torch.ones(args.n_particles, requires_grad=False).type_as(z) / args.n_particles
    weight_half = torch.ones(args.n_particles // 2, requires_grad=False).type_as(z) / (args.n_particles / 2)

    summary(μ_decoder, (decoder_z_features, 1, 1))

    losses = []


    sinkhorn_divergence_obj = SamplesLoss(loss="sinkhorn", p=p, blur=blur_loss, backend=backend, scaling=scaling,
                                          debias=True)
    sinkhorn_divergence_con = SamplesLoss(loss="sinkhorn", p=p, blur=blur_constraint, backend=backend, scaling=scaling,
                                          debias=True)
    i = 0
    time_SiNG = 0
    for epoch in range(args.epochs):
        for data in tqdm(loader):
            time_start = time.time()
            x_real = data[0].to(device).view(args.batch_size, -1)
            # train the decoder with natural gradient
            # 0. construct the discrete approximate by sampling
            with torch.autograd.no_grad():
                μ_before = μ_decoder(z.normal_(0, 1)).view(args.batch_size, -1)
            # z.normal_(0, 1)
            for i_jko in range(jko_steps):
                optimizerμ.zero_grad()
                μ = μ_decoder(z).view(args.batch_size, -1)
                loss_jko = sinkhorn_divergence_obj(weight, μ, weight, x_real) \
                           + η_g * sinkhorn_divergence_con(weight, μ, weight, μ_before)
                loss_jko.backward()
                optimizerμ.step()

            time_SiNG += time.time() - time_start
            print(time_SiNG)
            with torch.autograd.no_grad():
                μ_after = μ_decoder(z).view(args.batch_size, -1)
                S_delta = sinkhorn_divergence_con(weight, μ_before, weight, μ_after).item()

            with torch.autograd.no_grad():
                φ_μ = μ_decoder(z).view(args.batch_size, -1)
                loss_print = sinkhorn_divergence_obj(weight, φ_μ, weight, x_real).item()

            writer.add_scalar('generative model loss_blur_{} constraint_blur_{} step_{} jko_steps_{} ng/loss'
                              .format(blur_loss, blur_constraint, η_g, jko_steps), loss_print, i)
            writer.add_scalar('generative model loss_blur_{} constraint_blur_{} step_{} jko_steps_{} ng/time'
                              .format(blur_loss, blur_constraint, η_g, jko_steps), time_SiNG, i)
            writer.add_scalar('generative model loss_blur_{} constraint_blur_{} step_{} jko_steps_{} ng/S_delta'
                              .format(blur_loss, blur_constraint, η_g, jko_steps), S_delta,
                              i)
            writer.flush()

            nrow = int(math.sqrt(args.n_observe))
            vutils.save_image(μ_decoder(z_observe), '{}/x_{}.png'.format(args.outf, i), normalize=True,
                                  nrow=nrow)
            i += 1
        torch.save(μ_decoder.state_dict(), '{}/μ_decoder_{}.pt'.format(args.modelf, epoch))
