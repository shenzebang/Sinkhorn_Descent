import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from geomloss import SamplesLoss

from utils import base_module

IMAGE_SIZE = 32
n_channels = 3
η = 1
p = 2
blur = 5
scaling = .95
noise_level = 0.1
d = n_channels*IMAGE_SIZE*IMAGE_SIZE
d_feature = 32
c_shape = [-1, n_channels, IMAGE_SIZE, IMAGE_SIZE]
batch_size = 10000

dataset = dset.CelebA(root='celeba', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(IMAGE_SIZE),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2, drop_last=True)
modelf = 'model/celebrity'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

c_encoder = base_module.Encoder(IMAGE_SIZE, n_channels, k=d_feature).cuda()
c_encoder.load_state_dict(torch.load('{}/c_encoder_{}.pt'.format(modelf, 202)))
sinkhorn_divergence = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend="online", scaling=scaling)
x_real_weight = torch.ones(batch_size, requires_grad=False).cuda() / batch_size

for i, data in enumerate(dataloader):
    with torch.autograd.no_grad():
        if i<1:
            φ_1 = c_encoder(data[0].to(device)).squeeze()
            continue
        φ_2 = c_encoder(data[0].to(device)).squeeze()
        # print(φ_1.shape)
        print(sinkhorn_divergence(x_real_weight, φ_1, x_real_weight, φ_2))
        φ_1 = φ_2