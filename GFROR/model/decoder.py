# Adapted from https://github.com/lwneal/counterfactual-open-set/blob/master/generativeopenset/network_definitions.py
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_size=100):
        super().__init__()

        self.latent_size = latent_size
        self.linear = nn.Linear(latent_size, 512*2*2, bias=False)

        self.block1_in = nn.ConvTranspose2d(latent_size,512,1,1,0,bias=False)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(512,512,4,2,1,bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
        )

        self.block2_in = nn.ConvTranspose2d(latent_size,512,1,1,0,bias=False)
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1,bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        )

        self.block3_in = nn.ConvTranspose2d(latent_size,256,1,1,0,bias=False)
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1,bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
        )

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(128,3,4,2,1),
            nn.Sigmoid()
        )
        


    def forward(self, x, in_scale=1):

        if in_scale <= 1:
            out = self.linear(x).reshape(-1,512,2,2)

        if in_scale == 2:
            out = out.view(-1, self.latent_size,in_scale,in_scale)
            out = self.block1_in(out)
        if in_scale <= 2:
            out = self.block1(out)

        if in_scale == 4:
            out = out.view(-1, self.latent_size,in_scale,in_scale)
            out = self.block2_in(out)
        if in_scale <= 4:
            out = self.block2(out)

        if in_scale == 8:
            out = out.view(-1, self.latent_size,in_scale,in_scale)
            out = self.block3_in(out)
        if in_scale <= 8:
            out = self.block3(out)

        return self.block4(out)




class Decoder320(nn.Module):
    def __init__(self, latent_size=512):
        super().__init__()
        self.linear = nn.Linear(latent_size, 512 * 20 * 20)
        
        self.deconv = nn.Sequential(
            # 20x20 -> 40x40
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 40x40 -> 80x80
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 80x80 -> 160x160
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 160x160 -> 320x320
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh() # Ou Sigmoid, dependendo do seu pré-processamento
        )

    def forward(self, z):
        out = self.linear(z)
        out = out.view(out.shape[0], 512, 20, 20)
        return self.deconv(out)