import struct

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

import utils

#seeds
torch.manual_seed(42)
np.random.seed(42)

#loading data
x_train, y = utils.load_mnist()

print("Data shape:", x_train.shape, y.shape)
x_train = utils.rescale_data(x_train)
len_data = (x_train.shape[0])


#Initializing variables according to DCGAN paper recommendations
lr = 0.0002
b1 = 0.5
nz = 100
bs = 128

class Reshape(nn.Module):
    def __init__(self, C, W, H):
        super(Reshape, self).__init__()
        self.C = C
        self.W = W
        self.H = H
        
    def forward(self, X):
        # print(X)
        return X.view(-1, self.C, self.W, self.H)

class Generator(nn.Module):
    def __init__(self, type_='convT'):
        super(Generator, self).__init__()
        self.type = type_
        lrelu = nn.LeakyReLU(0.2, inplace=True)
        reshape = Reshape(256, 7, 7)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, stride=1, bias = False),
            nn.BatchNorm2d(512),
            lrelu,
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding = 1, bias = False),
            # self.upscale(2, 512, 256, 3),
            nn.BatchNorm2d(256),
            lrelu,
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding = 1, bias = False),
            # self.upscale(2, 256, 128, 3),
            nn.BatchNorm2d(128),
            lrelu,
            # # nn.ConvTranspose2d(128, 1, 4, stride=2, padding = 1, bias = False),
            self.upscale(2, 128, 1, 3),
            nn.Tanh()
        )
        self.main_convT =  nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, stride=1, bias = False),
            nn.BatchNorm2d(512),
            lrelu,
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            lrelu,
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding = 2, bias = False),
            nn.BatchNorm2d(128),
            lrelu,
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding = 1, bias = False),
            nn.Tanh()
        )
        self.main_linear = nn.Sequential(
            nn.Linear(100, 7*7*256, bias=False),
            reshape,
            nn.BatchNorm2d(256),
            lrelu,
            self.upscale(2, 256, 128, 3),
            nn.BatchNorm2d(128),
            lrelu,
            self.upscale(2, 128, 1, 3),
            nn.Tanh()
        )
    def forward(self, x):
        if self.type == 'convT':
            return self.main_convT(x)
        elif self.type == 'linear':
            return self.main_linear(x)
        else:
            return self.main(x)
    
    def upscale(self, factor, in_C, out_C, k_size, padding=0):
        layers = [
            nn.Conv2d(in_C, out_C * (factor ** 2), kernel_size=k_size, padding=(k_size // factor) if not padding else padding),
            nn.PixelShuffle(factor)
        ]
        return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.main = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1, bias=False),
        lrelu,
        nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        lrelu,
        nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        lrelu,
        nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        lrelu,
        torch.nn.Flatten(),
        nn.Linear(2048, 1),
        nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x)

netG = Generator('convT').cuda()
netD = Discriminator().cuda()

netG.apply(utils.weights_init)
netD.apply(utils.weights_init)

# summary(netG, input_size=(1, nz))
# summary(netG, input_size=(nz, 1, 1))
# summary(netD, input_size=(1, 28, 28))

# exit()

# VISUALIZE SAMPLE OUTPUT

# sample = np.random.randn(100)
# sample = torch.Tensor(sample).view(1, -1, 1, 1).cuda()
# y_hat = netG(sample)
# show_img(y_hat.detach().cpu().squeeze().numpy())

# Training

def sample_noise(bs):
    global netG
    if netG.type == 'linear':
        return torch.randn(bs, nz).cuda()
    else:
        return torch.randn(bs, nz, 1, 1).cuda()

def train_GAN(steps = 1, training_iter= 1000):
    running_count = 0
    epoch = 0 
    def generator_loss(output):
        return -torch.sum(torch.log(output)) / output.shape[0]
    x_sub = x_train[:]
    y_sub = y[:]

    opti_netD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(b1, 0.999))
    opti_netG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(b1, 0.999))
    criterion = nn.BCELoss()

    acc_loss_d = 0
    acc_loss_g = 0
    sample = sample_noise(16)
    
    for i in range(training_iter):
        running_count += bs
        if running_count > len_data:
            epoch += 1
            running_count = 0
        if i % 200 == 0:
            utils.plot_grid(netG, sample)
            plt.pause(0.00005)
        for s in range(steps):
            opti_netD.zero_grad()
            x_true = torch.Tensor(x_sub[np.random.randint(0, len(x_sub), size=bs),:,:]).view(-1, 1, 28, 28).cuda()
            y_true = torch.ones(x_true.shape[0]).cuda()
            x_false = sample_noise(bs)
            y_false = torch.zeros(x_false.shape[0]).cuda()
            
            out_true = x_true
            fake = netG(x_false)
            res_true = netD(out_true).squeeze(1)
            res_false = netD(fake.detach()).squeeze(1)
            
            loss_D = criterion(res_true, y_true) + criterion(res_false, y_false)
            loss_D.backward()
            acc_loss_d += loss_D
            opti_netD.step()
        
        opti_netG.zero_grad()
        x_false = sample_noise(bs)
        # out_false = netG(x_false)
        res_false = netD(fake).squeeze(1)
        # print(res_false)
        loss_G = criterion(res_false, y_true)
        loss_G.backward()
        acc_loss_g += loss_G
        opti_netG.step()
        
        if i > 0 and i % 20:
            print(f"Epoch {epoch} Loss D {acc_loss_d / 20:8.3}    Loss G {acc_loss_g / 20:7.3}")
            acc_loss_d = 0
            acc_loss_g = 0
        if epoch > 10:
            break
        # true = sum(y_hat.max(1)[1].detach().cpu().numpy() == batch_y.detach().cpu().numpy())
        # print(f"batch accuracy: {round(true / bs, 2)}")
        # break

train_GAN(1, 10000)
