import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')

from geomloss import SamplesLoss
from utils import base_module_high_dim_encoder
from utils import base_module_ot_gan
from utils.sinkhorn_util import potential_operator_grad
from torchsummary import summary
import time
import math
from tqdm import tqdm
import torch.utils.data as Data
from utils.conjugate_gradient import conjugate_gradients, set_flat_params_to, get_flat_params_from
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    gan_name = 'Sinkhorn_GAN'
    algorithm_name = 'SiNG'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='celebA', help='dataset to be used')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=3000)
    parser.add_argument('--n_particles', type=int, default=3000)
    parser.add_argument('--n_observe', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--niterD', type=int, default=1, help='no. updates of D per update of G')
    parser.add_argument('--niterG', type=int, default=3, help='no. updates of G per update of D')
    parser.add_argument('--decoder_z_features', type=int, default=64)
    parser.add_argument('--lr_encoder', type=float, default=1e-3, help='learning rate of c_encoder')
    parser.add_argument('--eta', type=float, default=30, help='step size of the natural gradient update')
    parser.add_argument('--damping', type=float, default=.05, help='damping parameter of natural gradient')
    parser.add_argument('--max_sd', type=float, default=.1, help='maximum allowed sd parameter of natural gradient')
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
    η_g = args.eta
    p = 2
    blur_loss = 10
    blur_constraint = 10
    potential_iter = 10
    cg_iter = 20
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



    # initialize the encoder
    c_encoder = base_module_high_dim_encoder.Encoder(IMAGE_SIZE, n_channels).cuda()
    c_encoder.apply(base_module_high_dim_encoder.weights_init)
    optimizerC = optim.Adam(c_encoder.parameters(), lr=args.lr_encoder, betas=(0.5, 0.9), amsgrad=True)

    summary(c_encoder, (n_channels, IMAGE_SIZE, IMAGE_SIZE))
    summary(μ_decoder, (decoder_z_features, 1, 1))

    losses = []

    sinkhorn_divergence = SamplesLoss(loss="sinkhorn", p=p, blur=blur_loss, backend=backend, scaling=scaling, debias=True)
    potential_operator = SamplesLoss(loss="sinkhorn", p=p, blur=blur_constraint, potentials=True, debias=False, backend=backend,
                                     scaling=scaling)
    i = 0
    for epoch in range(args.epochs):
        time_start = time.time()
        loss_accumulation = 0
        G_count = 0
        for data in tqdm(loader):
            x_real = data[0].to(device)
            if i % (args.niterG + 1) == 0:
                μ_detach = μ_decoder(z.normal_(0, 1)).detach()
                # --- train the encoder of ground cost:
                # ---   optimizing on the ground space endowed with cost \|φ(x) - φ(y)\|^2 is equivalent to
                # ---   optimizing on the feature space with cost \|x - y\|^2.
                optimizerC.zero_grad()
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

                # del φ_x_real_1, φ_x_real_2, φ_μ_1, φ_μ_2, negative_loss, μ_detach
                del negative_loss, μ_detach
                torch.cuda.empty_cache()
            else:
                G_count += 1
                # train the decoder with natural gradient
                # 0. construct the discrete approximate by sampling
                with torch.autograd.no_grad():
                    φ_x_real = c_encoder(x_real)
                φ_μ = c_encoder(μ_decoder(z.normal_(0, 1)))
                # 1. compute the gradient of the free energy
                loss_G = sinkhorn_divergence(weight, φ_μ, weight, φ_x_real)
                # f_energy_αβ, _ = potential_operator(μ_weight, φ_μ,  x_real_weight, φ_x_real)
                # f_energy_αα, _ = potential_operator(μ_weight, φ_μ, μ_weight, φ_μ)
                grads = torch.autograd.grad(loss_G, μ_decoder.parameters())
                loss_print = loss_G.item()
                # loss_accumulation += loss_G.item()
                # print("epoch {0}, inner loop, μ loss {1:9.3e}".format(epoch, loss_G.item()))
                loss_grad = torch.cat([grad.view(-1) for grad in grads])
                del φ_μ, x_real, φ_x_real, loss_G
                torch.cuda.empty_cache()
                # 2. compute the Hessian vector product
                with torch.autograd.no_grad():
                    φ_μ_before = c_encoder(μ_decoder(z.normal_(0, 1)))
                    f_metric_αα_value, g_metric_αα_value = potential_operator(weight, φ_μ_before, weight, φ_μ_before)
                    # print((f_metric_αα_value-g_metric_αα_value).norm())
                def mvp(v):
                    φ_μ = c_encoder(μ_decoder(z))
                    # φ_μ = (μ_decoder(z)).view([args.n_particles, -1])
                    # a. compute the sinkhorn potential (value only, no grad_fn)
                    # f_metric_αα_value, _ = potential_operator(μ_weight, φ_μ, μ_weight, φ_μ)
                    # b. compute the sinkhorn potential (including grad_fn)
                    f_metric_α_α_grad1, g_metric_α_α_grad1 = potential_operator_grad(
                        weight,
                        φ_μ,
                        weight,
                        φ_μ.detach(),
                        f_metric_αα_value, g_metric_αα_value,  blur_constraint, p, backend=backend, niter=potential_iter
                    )
                    f_metric_α_α_grad, g_metric_α_α_grad = potential_operator_grad(
                        weight,
                        φ_μ,
                        weight,
                        φ_μ.detach(),
                        f_metric_α_α_grad1.detach(), g_metric_α_α_grad1.detach(), blur_constraint, p, backend=backend, niter=potential_iter
                    )
                    f_metric_αα_grad, _ = potential_operator_grad(
                        weight,
                        φ_μ,
                        weight,
                        φ_μ,
                        f_metric_α_α_grad.detach(), g_metric_α_α_grad1.detach(), blur_constraint, p, backend=backend, niter=potential_iter
                    )
                    # DUBUG: compute the accuracy of sinkhorn iterate
                    # print('DEBUG in cg')
                    # print(torch.norm(f_metric_α_α_grad - g_metric_α_α_grad)/ (torch.norm(f_metric_α_α_grad)+1e-7))
                    # print(torch.norm(f_metric_α_α_grad - f_metric_αα_grad) / (torch.norm(f_metric_α_α_grad)+1e-7))
                    # print(torch.norm(f_metric_α_α_grad - f_metric_α_α_grad1)/ (torch.norm(f_metric_α_α_grad)+1e-7))
                    # c. compute the Hvp α_ → α
                    sd = torch.sum(f_metric_α_α_grad + g_metric_α_α_grad - f_metric_αα_grad)
                    # there is a missing term g_metric_ββ_grad which is omitted as it does not influence the gradient computation
                    grads = torch.autograd.grad(sd, μ_decoder.parameters(), create_graph=True)
                    flat_grad_sd = torch.cat([grad.view(-1) for grad in grads])

                    sd_v = (flat_grad_sd * v).sum()
                    grads = torch.autograd.grad(sd_v, μ_decoder.parameters())
                    flat_grad_grad_sd = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

                    return flat_grad_grad_sd + v * args.damping

                stepdir = conjugate_gradients(mvp, -loss_grad, cg_iter)
                relative_CG_residual = torch.norm(mvp(stepdir)+loss_grad)/torch.norm(loss_grad)
                inner_product_between_CG_and_G = stepdir.dot(-loss_grad)/torch.norm(stepdir)/torch.norm(loss_grad)
                shs = 0.5 * (stepdir * mvp(stepdir)).sum(0, keepdim=True)
                lm = torch.sqrt(shs / args.max_sd)
                # print(lm[0])
                prev_params = get_flat_params_from(μ_decoder)
                xnew = prev_params + η_g / lm[0] * stepdir
                set_flat_params_to(μ_decoder, xnew)

                with torch.autograd.no_grad():
                    φ_μ_after = c_encoder(μ_decoder(z))
                    S_delta = sinkhorn_divergence(weight, φ_μ_before, weight, φ_μ_after)



                writer.add_scalar('GAN loss_blur_{} constraint_blur_{} step_{} potential_iter_{} ng/loss'
                                  .format(blur_loss, blur_constraint, η_g, potential_iter), loss_print, i)
                writer.add_scalar('GAN loss_blur_{} constraint_blur_{} step_{} potential_iter_{} ng/CG_residual'
                                  .format(blur_loss, blur_constraint, η_g, potential_iter), relative_CG_residual,
                                  i)
                writer.add_scalar('GAN loss_blur_{} constraint_blur_{} step_{} potential_iter_{} ng/S_delta'
                                  .format(blur_loss, blur_constraint, η_g, potential_iter), S_delta,
                                  i)
                writer.add_scalar('GAN loss_blur_{} constraint_blur_{} step_{} potential_iter_{} ng/CG_dot_G'
                                  .format(blur_loss, blur_constraint, η_g, potential_iter),
                                  inner_product_between_CG_and_G, i)
                writer.flush()

                del loss_grad, stepdir, prev_params, xnew, φ_μ_after, S_delta
                torch.cuda.empty_cache()

                nrow = int(math.sqrt(args.n_observe))
                if i % 10 == 0:
                    vutils.save_image(μ_decoder(z_observe), '{}/x_{}.png'.format(args.outf, i), normalize=True,
                                  nrow=nrow)

            i += 1

        torch.save(c_encoder.state_dict(), '{}/c_encoder_{}.pt'.format(args.modelf, epoch))
        torch.save(μ_decoder.state_dict(), '{}/μ_decoder_{}.pt'.format(args.modelf, epoch))
