import torch
from torch import nn, optim, FloatTensor, LongTensor
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt

devicename = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(devicename)
print(f'Training GAN Model on {device} with MNIST Data\n{"=" * 44}')

opt = {
    'batch_size': 128,
    'label_size': 10,
    'initial_vector_size': 100,
    'num_epoch': 200
}

transform=transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
])

mnist_train = datasets.MNIST(root="MNIST/",
                             train=True,
                             download=True,
                             transform=transform)
# mnist_test = datasets.MNIST(root="MNIST/",
#                              train=True,
#                              download=True,
#                              transform=transform)

train_loader = DataLoader(dataset=mnist_train,
                        batch_size=opt['batch_size'],
                        shuffle=True,
                        drop_last=True)
# test_loader = DataLoader(dataset=mnist_test,
#                        batch_size=opt['batch_size'],
#                        shuffle=True)
# utils
def InitialVector():
  return torch.randn(opt['batch_size'], opt['initial_vector_size'], device=device)

def RandomLabel():
  return Variable(torch.LongTensor(np.random.randint(0, opt['label_size'], opt['batch_size']))).to(device)

def make_one_hot(labels):
    one_hot = torch.FloatTensor(opt['batch_size'], opt['label_size']).zero_().to(device)
    target = one_hot.scatter_(1, labels.unsqueeze(1), 1)
    target = Variable(target)
    return target

# model - Generator and Discriminator
class Generator(nn.Module):
  def __init__(self, d=128):
    super().__init__()
    self.main = nn.Sequential(
        nn.ConvTranspose2d(100, d*8, 4, 1, 0),
        nn.BatchNorm2d(d*8),
        nn.ReLU(True),

        nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
        nn.BatchNorm2d(d*4),
        nn.ReLU(True),

        nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
        nn.BatchNorm2d(d*2),
        nn.ReLU(True),

        nn.ConvTranspose2d(d*2, d, 4, 2, 1),
        nn.BatchNorm2d(d),
        nn.ReLU(True),

        nn.ConvTranspose2d(d, 1, 4, 2, 1),
        nn.Tanh()
    )
  def forward(self, z):
    z = z.reshape([z.size(0), 100, 1, 1])
    out = self.main(z)
    return out

class Discriminator(nn.Module):
  def __init__(self, d=128):
    super().__init__()
    self.main = nn.Sequential(
        # 1 * 28 * 28
        # nn.Conv2d(1, d, kernel_size=4, stride=2, padding=1),   # kernel_size, stride, padding
        nn.Conv2d(1, d, 4, 2, 1),   # kernel_size, stride, padding
        nn.LeakyReLU(0.2, inplace=True),
        # 128 * 32 * 32
        nn.Conv2d(d, d*2, 4, 2, 1),
        nn.BatchNorm2d(d*2),
        nn.LeakyReLU(0.2, inplace=True),
        # 256 * 16 * 16
        nn.Conv2d(d*2, d*4, 4, 2, 1),
        nn.BatchNorm2d(d*4),
        nn.LeakyReLU(0.2, inplace=True),
        # 512 * 8 * 8
        nn.Conv2d(d*4, d*8, 4, 2, 1),
        nn.BatchNorm2d(d*8),
        nn.LeakyReLU(0.2, inplace=True),
        #1024 * 4 * 4
        nn.Conv2d(d*8, 1, 4, 1, 0),
        nn.Sigmoid()
    )
  def forward(self, x):
    x = self.main(x)
    x = x.reshape([x.size(0), 1])
    return x

generator = Generator().to(device)
discriminator = Discriminator().to(device)
# optimizers
optimizer_g = optim.Adam(generator.parameters(), lr = 0.0001)
optimizer_d = optim.Adam(discriminator.parameters(), lr = 0.0001)

# cost/loss function
criterion = nn.BCELoss()

def imgshow(epoch):
  generator.eval()
  z = Variable(torch.randn(opt['label_size'] ** 2, opt['label_size'] ** 2)).to(device)

  images = generator(z)
  images = images.cpu().detach()

  grid = make_grid(images, nrow=opt['label_size'], normalize=True)
  save_image(grid, "result.%d.png"%(epoch+1))
  fig, ax = plt.subplots(figsize=(opt['label_size'],opt['label_size']))
  ax.imshow(grid.permute(1, 2, 0).data, cmap='binary')
  ax.axis('off')
  plt.show()

def sample_z(batch_size = 1, d_noise=100):
    return torch.randn(batch_size, d_noise, device=device)

# training
def train(epoch):
  print('starting {}/{}'.format(epoch, opt['num_epoch']))
  generator.train()
  discriminator.train()
  avg_loss = [0, 0]

  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)

    # maximize V(D,G) on D's sight
    p_real = discriminator(data)
    p_fake = discriminator(generator(sample_z(opt['batch_size'], d_noise=100)))

    loss_real = -1 * torch.log(p_real)   # -1 for gradient ascending
    loss_fake = -1 * torch.log(1.-p_fake) # -1 for gradient ascending
    loss_d = (loss_real + loss_fake).mean()

    loss_d.backward()
    optimizer_d.step()

    # minimize V(D, G) on G's sight
    p_fake = discriminator(generator(sample_z(opt['batch_size'], d_noise=100)))
    loss_g = -1 * torch.log(p_fake).mean()
    loss_g.backward()
    
    optimizer_g.step()

    optimizer_g.zero_grad()
    optimizer_d.zero_grad()
    avg_loss[0] += loss_g
    avg_loss[1] += loss_d

  avg_loss[0] /= len(train_loader)
  avg_loss[1] /= len(train_loader)  
  return avg_loss

# main
if __name__=="__main__":
  for epoch in range(opt['num_epoch']):
    avg_loss = train(epoch)
    print(f'epoch {epoch} result: d_loss: {avg_loss[1]} g_loss: {avg_loss[0]}')
    if (epoch % 25 == 0):
      imgshow(epoch)