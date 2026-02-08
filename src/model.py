import torch
import torch.nn as nn

class Conv_VAE(nn.Module):
    def __init__(self):
        super(Conv_VAE, self).__init__()

        self.encoder = nn.Sequential()
        self.encoder.add_module('conv1', nn.Conv1d(in_channels=4, out_channels=16, kernel_size=2, stride=1))            
        self.encoder.add_module('relu1', nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.encoder.add_module('dropout_enc2', nn.Dropout1d(p=0.2))
      
        self.encoder.add_module('conv2', nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1))
        self.encoder.add_module('relu2', nn.LeakyReLU(negative_slope=0.1, inplace=True))
      
        self.encoder.add_module('conv3', nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1))
        self.encoder.add_module('relu3', nn.LeakyReLU(negative_slope=0.1, inplace=True))


        self._mu = nn.Linear(in_features=64, out_features=16)
        self._logvar = nn.Linear(in_features=64, out_features=16)

        self.decoder = nn.Sequential()
        self.decoder.add_module('tconv3', nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=2, stride=1))
        self.decoder.add_module('relu3', nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.decoder.add_module('dropout_dec3', nn.Dropout1d(p=0.2)) 
      
        self.decoder.add_module('tconv2', nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=2, stride=1))
        self.decoder.add_module('relu2',nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.decoder.add_module('dropout_dec2', nn.Dropout1d(p=0.2)) 

        self.decoder.add_module('tconv1', nn.ConvTranspose1d(in_channels=64, out_channels=4, kernel_size=2, stride=1))
        self.decoder.add_module('sigmoid1', nn.Sigmoid())

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        sampling = mu + (eps * std)
        return sampling

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)

        mu = self._mu(x)
        logvar = self._logvar(x)

        x = self.reparameterization(mu, logvar)

        x = x.view(-1, 16, 1)

        return self.decoder(x), mu, logvar
