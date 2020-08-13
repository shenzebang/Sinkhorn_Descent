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
        digit_weight_np = []
        digit_support_np = []
        digit_weight_torch = []
        digit_support_torch = []
        digit_index = np.where(label == digit)[0]
        digit_image = image[digit_index].view(-1, 28, 28).tolist()
        for digit_image_i in digit_image:
            digit_image_i = np.array(digit_image_i)
            support_index = np.where(digit_image_i > 0)
            support = np.array(support_index)/28
            support = np.transpose(support)
            weight = digit_image_i[support_index]
            weight = weight/sum(weight)
            digit_weight_np.append(weight)
            digit_support_np.append(support)
            digit_weight_torch.append(torch.FloatTensor(weight))
            digit_support_torch.append(torch.FloatTensor(support))
        # digit_image_support_index = np.where(digit_image > 0)
        # digit_image_support = digit_image_support_index/28
        # digit_image_support_list = digit_image_support.tolist()
        # digit_image_weight = digit_image[digit_image_support_index]
        # digit_image_weight = digit_image_weight/torch.sum(digit_image_weight, [1, 2])
        torch.save(digit_weight_np, '{}/mnist_{}_weight_np.pt'.format(dataf, digit))
        torch.save(digit_support_np, '{}/mnist_{}_support_np.pt'.format(dataf, digit))
        torch.save(digit_weight_torch, '{}/mnist_{}_weight_torch.pt'.format(dataf, digit))
        torch.save(digit_support_torch, '{}/mnist_{}_support_torch.pt'.format(dataf, digit))