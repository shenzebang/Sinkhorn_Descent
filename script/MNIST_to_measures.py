import argparse
import os
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from core.distribution import Distribution

script_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_path, '../data', 'mnist')
os.system('mkdir -p {}'.format(data_path))
dataset = dset.MNIST(root=data_path, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
                                         shuffle=True, num_workers=8, drop_last=True)

data = next(iter(dataloader))
images = data[0]
label = data[1].numpy()

for digit in range(10):
    digit_list = []
    digit_index = np.where(label == digit)[0]
    digit_images = images[digit_index].view(-1, 28, 28).tolist()
    for digit_image in digit_images:
        digit_image = np.array(digit_image)
        support_index = np.where(digit_image > 0)
        support = 1 - np.array(support_index)/28
        support = np.transpose(support)
        weights = digit_image[support_index]
        weights = weights/sum(weights)
        digit_list.append(Distribution(torch.FloatTensor(support), torch.FloatTensor(weights)))
    torch.save(digit_list, '{}/digit_{}.pt'.format(data_path, digit))
