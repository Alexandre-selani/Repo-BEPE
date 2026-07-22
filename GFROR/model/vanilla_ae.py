import torch
import torch.nn as nn
from model.encoder import Encoder,Encoder320
from model.decoder import Decoder,Decoder320

class VanillaAE(nn.Module):
    def __init__(self, latent_size=100):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):

        return self.decoder(self.encoder(x))



class VanillaAE320(nn.Module):
    def __init__(self, latent_size=1000):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = Encoder320(latent_size)
        self.decoder = Decoder320(latent_size)

    def forward(self, x):

        return self.decoder(self.encoder(x))
