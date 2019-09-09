import torch
import torch.nn as nn
from torch import optim
from torch.autograd.variable import Variable

from torchvision import transforms
import torchvision.datasets as datasets

from discriminator import Discriminator
from generator import Generator

# load dataset
# normalizing images between -1 to 1
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ]))

# creating dataloader
data_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=100, shuffle=True)

# Num batches
no_batches = len(data_loader)

# generating a 1-d vector of gaussian sampled random values
rand_noise = Variable(torch.randn(size, 100))

# defining the discriminator
discriminator = Discriminator()

# defining the generator
generator = Generator()

# loss
criterion = nn.BCELoss()

# optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# training
