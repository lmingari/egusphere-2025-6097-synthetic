import torch
import torch.nn as nn
import torch.nn.functional as F

##############
### Models ###
##############
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        ### Encoder ###
        self.encoder = nn.Sequential(
                nn.Conv2d(1,16,kernel_size=3, stride=2, padding=1), # N,16,51,61
                nn.ReLU(True),
                nn.Conv2d(16,32,kernel_size=3, stride=2, padding=1), #N,32,26,31
                nn.ReLU(True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), #N,64,13,16
                nn.ReLU(True),
                nn.Flatten(),
                nn.Linear(64*13*16,latent_dim)
                )

        ### Decoder ###
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64*13*16),
                nn.ReLU(True),
                nn.Unflatten(1, (64,13,16)),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1,0)),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(0,0)),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(0,0)),
                nn.ReLU(True)
                )
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def encode(self,x):
        return self.encoder(x)

    def decode(self,z):
        return self.decoder(z)

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim, upsampling='transpose'):
        super().__init__()

        ### Encoder ###
        self.encoder = nn.Sequential(
                nn.Conv2d(1,16,kernel_size=3, stride=2, padding=1), # N,16,51,61
                nn.ReLU(True),
                nn.Conv2d(16,32,kernel_size=3, stride=2, padding=1), #N,32,26,31
                nn.ReLU(True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), #N,64,13,16
                nn.ReLU(True),
                nn.Flatten(),
                )

        self.fc_mu     = nn.Linear(64*13*16,latent_dim)
        self.fc_logvar = nn.Linear(64*13*16,latent_dim)

        ### Decoder ###
        if upsampling == 'transpose':
            self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 64*13*16),
                    nn.ReLU(True),
                    nn.Unflatten(1, (64,13,16)),
                    nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1,0)),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(0,0)),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(0,0)),
                    nn.ReLU(True)
                    )
        elif upsampling == 'bilinear':
            self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 64*13*16),
                    nn.ReLU(True),
                    nn.Unflatten(1, (64,13,16)),
                    nn.Upsample(size=(26,31), mode='bilinear', align_corners=False),
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.ReLU(True),
                    nn.Upsample(size=(51,61), mode='bilinear', align_corners=False),
                    nn.Conv2d(32, 16, kernel_size=3, padding=1),
                    nn.ReLU(True),
                    nn.Upsample(size=(101,121), mode='bilinear', align_corners=False),
                    nn.Conv2d(16, 1, kernel_size=3, padding=1),
                    nn.ReLU(True)
                    )
        else:
            raise ValueError("upsampling must be 'transpose' or 'bilinear'")
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar

    def encode(self,x):
        h = self.encoder(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self,z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

### VAE with kernel 5x5
class VAE5(nn.Module):
    def __init__(self, latent_dim=16, input_channels=1):
        super().__init__()

        ### Encoder ###
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, latent_dim, kernel_size=5, stride=2, padding=2),  # -> 16, H/2, W/2
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),              # -> 32, H/4, W/4
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),              # -> 64, H/8, W/8
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        self.fc_mu     = nn.Linear(64*13*16,latent_dim)
        self.fc_logvar = nn.Linear(64*13*16,latent_dim)

        ### Decoder ###
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64*13*16),
                nn.ReLU(True),
                nn.Unflatten(1, (64,13,16)),
                nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=(1,0)),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=0),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=0),
                nn.ReLU(True)
                )

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar

    def encode(self,x):
        h = self.encoder(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self,z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

##################
### Criterions ###
##################
class VAELoss(nn.Module):
    """
    Loss function for a Variational Autoencoder (VAE).

    Args:
        beta (float): weight for KL divergence (β-VAE)
        reduction (str): 'mean' or 'sum'
    """
    def __init__(self, beta=1.0, reduction='sum'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def reconstruction_loss(self, recon_x, x):
        return F.mse_loss(recon_x, x, reduction=self.reduction)

    def kl_divergence(self, mu, logvar):
        # KL Divergence: D_KL(N(mu, σ) || N(0, I))
        # = -0.5 * sum(1 + log(σ^2) - μ^2 - σ^2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        if self.reduction == 'mean':
            kl = kl.mean()
        else:
            kl = kl.sum()

        return kl

    def forward(self, recon_x, x, mu, logvar):
        recon = self.reconstruction_loss(recon_x, x)
        kl = self.kl_divergence(mu, logvar)
        loss = recon + self.beta * kl
        return loss, recon, kl