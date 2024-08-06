import torch
import torch.nn as nn

class EdgeEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(EdgeEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, latent_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class EdgeDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(EdgeDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Assuming input images are normalized between 0 and 1
        )
    
    def forward(self, z):
        return self.decoder(z)

class EdgeAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(EdgeAutoencoder, self).__init__()
        self.encoder = EdgeEncoder(latent_dim)
        self.decoder = EdgeDecoder(latent_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
