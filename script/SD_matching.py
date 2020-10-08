import argparse
import torch
# torch.set_default_tensor_type(torch.DoubleTensor)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from PIL import Image

import numpy as np

import os
import sys
from core.sinkhorn_descent import SinkhornDescent
from core.distribution import Distribution
from utils.plot_utils import plot2




def distribution_matching(target_distribution, source_distribution, save_path, step_size=1, nit=201, frequency_loss_evaluation=20, backend="pytorch", image_size=None):
    # ==============================================================================================================
    #   compute Sinkhorn barycenter via Sinkhorn Descent
    # ==============================================================================================================
    sd = SinkhornDescent([target_distribution], source_distribution, step_size, backend=backend, blur=0.01)
    fig = plt.figure()
    ax = fig.gca()
    plt.clf()
    # ==============================================================================================================
    #   store the losses for plotting
    # ==============================================================================================================
    loss_vector = np.zeros(np.int(nit/frequency_loss_evaluation)+1)
    iter_vector = np.arange(0, nit, frequency_loss_evaluation)+1
    if image_size is None:
        image_size = [100, 100]
    plt.axes().set_aspect(float(image_size[0])/image_size[1])
    for i in range(nit):
        sd.step()
        if i % frequency_loss_evaluation == 0:
            barycenter_plot = sd.barycenter.support.detach().cpu()
            barycenter_weight = sd.barycenter.weights.cpu()
            plot2(barycenter_plot, barycenter_weight, binx=image_size[0], biny=image_size[1])
            plt.axis('off')
            plt.savefig(os.path.join(save_path, 'barycenter_{}.png'.format(i)), bbox_inches='tight')
            loss = sd.evaluate()
            loss_vector[np.int(i / frequency_loss_evaluation)] = loss
            print("iteration {}, loss {}".format(i, loss))


    # np.savetxt(os.path.join(plot_save_path, "SD_matching_nparticle{}".format(m)), np.vstack((iter_vector, loss_vector)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=100)
    parser.add_argument('--target_image', default='cheetah.jpg', help='image describing the target distribution')
    parser.add_argument('--support_size', type=int, default=8000)
    parser.add_argument('--step_size', type=float, default=1, help='learning rate')

    args = parser.parse_args()

    script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_path, '../data', 'matching', args.target_image)
    save_path = os.path.join(script_path, '../out', 'matching')
    os.system('mkdir -p {}'.format(save_path))

    # load and resize image
    img = Image.open(data_path)
    height = img.height
    width = img.width
    aspect = float(height)/width
    pix = np.array(img)
    pix = 255 - pix

    # visualize and save the resized original image
    try:
        imgplot = plt.imshow(img)
        plt.savefig(os.path.join(save_path, 'original.png'))
        plt.pause(0.1)
    finally:
        pass

    # create a meshgrid and interpret the image as a probability distribution on it
    x_ls = torch.linspace(0, 1, steps=pix.shape[0])
    y_ls = torch.linspace(0, pix.shape[1]/pix.shape[0], steps=pix.shape[1])
    X, Y = torch.meshgrid(x_ls, y_ls)
    X1 = X.reshape(-1)
    Y1 = Y.reshape(-1)
    y1 = []

    MX = 1

    weights = []
    pix_arr = pix[:, :, 0].reshape(-1)
    for value, x, y in zip(pix_arr, X1, Y1):
        if value > 50:
            y1.append(torch.tensor([y, MX - x]))
            weights.append(torch.tensor(value, dtype=torch.float32))

    nu1t = torch.stack(y1)
    w1 = torch.stack(weights).reshape((len(weights), 1))
    w1 = w1 / (torch.sum(w1, dim=0)[0])

    target_distribution = Distribution(nu1t.float().cuda(), w1.float().cuda().squeeze())

    # initialize the source_distribution
    dimension = 2
    initial_support = torch.rand(args.support_size, dimension, requires_grad=True, dtype=torch.float32).cuda()
    source_distribution = Distribution(initial_support)

    # plot_save_path = os.path.join(script_path, 'plot', 'matching')

    backend = "online"

    image_size = [int(aspect*args.image_size), args.image_size]
    distribution_matching(target_distribution=target_distribution,
                          source_distribution=source_distribution,
                          step_size=args.step_size,
                          image_size=image_size,
                          save_path=save_path,
                          backend=backend
                          )
