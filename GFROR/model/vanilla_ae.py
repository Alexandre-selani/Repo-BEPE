import torch
import torch.nn as nn
from model.encoder import Encoder,Encoder320,Encoder64,Encoder_eucalyptus
from model.decoder import Decoder,Decoder320,Decoder64,Decoder_eucalyptus

class VanillaAE(nn.Module):
    def __init__(self, latent_size=100):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):

        return self.decoder(self.encoder(x))



class VanillaAE64(nn.Module):
    """Autoencoder para TinyImageNet (entradas 64x64)."""

    def __init__(self, latent_size=512):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = Encoder64(latent_size)
        self.decoder = Decoder64(latent_size)

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



 
    
 
 

 
 
# ---------------------------------------------------------------------------
# VanillaAE
# ---------------------------------------------------------------------------
class VanillaAE_eucalyptus(nn.Module):
    def __init__(self, latent_size=512):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = Encoder_eucalyptus(latent_size)
        self.decoder = Decoder_eucalyptus(latent_size)
 
    def forward(self, x):
        return self.decoder(self.encoder(x))