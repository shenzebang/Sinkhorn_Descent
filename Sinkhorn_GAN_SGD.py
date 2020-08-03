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

    gan_name = 'Sinkhorn_GAN'
    algorithm_name = 'SGD'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='celebA', help='dataset to be used')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=3000)
    parser.add_argument('--n_particles', type=int, default=3000)
    parser.add_argument('--n_observe', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--niterD', type=int, default=1, help='no. updates of D per update of G')
    parser.add_argument('--niterG', type=int, default=2, help='no. updates of G per update of D')
    parser.add_argument('--decoder_z_features', type=int, default=64)
    parser.add_argument('--lr_encoder', type=float, default=1e-3, help='learning rate of c_encoder')
    parser.add_argument('--lr_decoder', type=float, default=1e-3, help='learning rate of μ_decoder')
    parser.add_argument('--scaling', type=float, default=.95, help='scaling parameter for the Geomloss package')
    parser.add_argument('--generator_backend', default='DC-GAN', help='NN model of the generator')
    args = parser.parse_args()

    cudnn.benchmark = True
    args.outf = "output/{}/{}/{}".format(gan_name, algorithm_name, args.dataset)
    args.modelf = "model/{}/{}/{}".format(gan_name, algorithm_name, args.dataset)
    args.dataf = "data_transform"

    if not os.path.exists(args.outf): os.makedirs(args.outf)
    if not os.path.exists(args.modelf): os.makedirs(args.modelf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(args.outf)
    IMAGE_SIZE = args.image_size
    p = 2
    blur_loss = 10
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
    optimizerμ = optim.Adam(μ_decoder.parameters(), lr=args.lr_decoder, betas=(0.5, 0.999), amsgrad=False)

    # initialize the encoder
    c_encoder = base_module_high_dim_encoder.Encoder(IMAGE_SIZE, n_channels).cuda()
    c_encoder.apply(base_module_high_dim_encoder.weights_init)
    optimizerC = optim.Adam(c_encoder.parameters(), lr=args.lr_encoder, betas=(0.5, 0.9), amsgrad=True)

    summary(c_encoder, (n_channels, IMAGE_SIZE, IMAGE_SIZE))
    summary(μ_decoder, (decoder_z_features, 1, 1))

    z = torch.cuda.FloatTensor(args.n_particles, decoder_z_features, 1, 1)
    z_observe = torch.cuda.FloatTensor(args.n_observe, decoder_z_features, 1, 1).normal_(0, 1)
    weight = torch.ones(args.n_particles, requires_grad=False).type_as(z) / args.n_particles
    weight_half = torch.ones(args.n_particles // 2, requires_grad=False).type_as(z) / (args.n_particles / 2)

    losses = []

    sinkhorn_divergence = SamplesLoss(loss="sinkhorn", p=p, blur=blur_loss, backend=backend, scaling=scaling, debias=True)
    i = 0
    for epoch in range(args.epochs):
        time_start = time.time()
        loss_accumulation = 0
        G_count = 0
        for data in tqdm(loader):
            x_real = data[0].to(device)
            if i % (args.niterG + 1) == 0:
                optimizerC.zero_grad()
                μ_detach = μ_decoder(z.normal_(0, 1)).detach()
                φ_x_real_1, φ_x_real_2 = c_encoder(x_real).chunk(2, dim=0)
                φ_μ_1, φ_μ_2 = c_encoder(μ_detach).chunk(2, dim=0)
                negative_loss = - sinkhorn_divergence(weight_half, φ_x_real_1, weight_half, φ_μ_1)
                negative_loss = - sinkhorn_divergence(weight_half, φ_x_real_2, weight_half, φ_μ_2) + negative_loss
                negative_loss = - sinkhorn_divergence(weight_half, φ_x_real_1, weight_half, φ_μ_2) + negative_loss
                negative_loss = - sinkhorn_divergence(weight_half, φ_x_real_2, weight_half, φ_μ_1) + negative_loss
                negative_loss =   sinkhorn_divergence(weight_half, φ_x_real_1, weight_half, φ_x_real_2) * 2 + negative_loss
                negative_loss =   sinkhorn_divergence(weight_half, φ_μ_1, weight_half, φ_μ_2) * 2 + negative_loss
                # φ_x_real = c_encoder(x_real)
                # φ_μ = c_encoder(μ_decoder(z.normal_(0, 1)))
                # negative_loss = -sinkhorn_divergence(μ_weight, φ_μ, x_real_weight, φ_x_real)
                negative_loss.backward()
                optimizerC.step()
                # for p in c_encoder.parameters():
                #     p.data.clamp_(-0.01, 0.01)
                del φ_x_real_1, φ_x_real_2, φ_μ_1, φ_μ_2
                del negative_loss, μ_detach
                torch.cuda.empty_cache()
            else:
                G_count += 1
                optimizerμ.zero_grad()
                with torch.autograd.no_grad():
                    φ_x_real = c_encoder(x_real)
                φ_μ = c_encoder(μ_decoder(z.normal_(0, 1)))
                loss_G = sinkhorn_divergence(weight, φ_μ, weight, φ_x_real)
                loss_G.backward()
                optimizerμ.step()

                writer.add_scalar('Sinkhorn GAN loss_blur_{} sg/loss'.format(blur_loss), loss_G.item(), i)
                writer.flush()

                nrow = int(math.sqrt(args.n_observe))
                if i % 10 == 0:
                    vutils.save_image(μ_decoder(z_observe), '{}/x_{}.png'.format(args.outf, i), normalize=True,
                                  nrow=nrow)

            i += 1

        torch.save(c_encoder.state_dict(), '{}/c_encoder_{}.pt'.format(args.modelf, epoch))
        torch.save(μ_decoder.state_dict(), '{}/μ_decoder_{}.pt'.format(args.modelf, epoch))
