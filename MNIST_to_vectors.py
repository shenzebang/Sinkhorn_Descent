import argparse
import os
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

if __name__ == '__main__':


    dataf = 'mnist'
    os.system('mkdir -p {}'.format(dataf))
    dataset = dset.MNIST(root='mnist', download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor()
                          ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
                                             shuffle=True, num_workers=8, drop_last=True)

    data = next(iter(dataloader))
    image = data[0]
    label = data[1].numpy()

    for digit in range(10):
        digit_weight_vec_np = []
        digit_weight_vec_torch = []
        digit_index = np.where(label == digit)[0]
        digit_image = image[digit_index].view(-1, 28, 28).tolist()
        for digit_image_i in digit_image:
            digit_image_i = np.array(digit_image_i).flatten()
            weight = digit_image_i/sum(digit_image_i)
            digit_weight_vec_np.append(weight)
            digit_weight_vec_torch.append(torch.FloatTensor(weight))
        torch.save(digit_weight_vec_np, '{}/mnist_{}_weight_vec_np.pt'.format(dataf, digit))
        torch.save(digit_weight_vec_torch, '{}/mnist_{}_weight_vec_torch.pt'.format(dataf, digit))