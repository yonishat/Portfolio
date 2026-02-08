import torch
import torch.nn as nn

# ==========================================
# 1. Standard Autoencoder (Baseline)
# ==========================================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('conv1', nn.Conv1d(in_channels=4, out_channels=16, kernel_size=2, stride=1))
        self.encoder.add_module('relu1', nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.encoder.add_module('conv2', nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1))
        self.encoder.add_module('relu2', nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.encoder.add_module('conv3', nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1))
        self.encoder.add_module('relu3', nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # Latent representation
        # Flattens 64 channels * 1 width -> 64 features
        self.latent = nn.Linear(in_features=64, out_features=16)

        # Decoder
        self.decoder = nn.Sequential()
        # Note: Input to decoder will be reshaped to (Batch, 16, 1)
        self.decoder.add_module('tconv3', nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=2, stride=1))
        self.decoder.add_module('relu3', nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.decoder.add_module('tconv2', nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=2, stride=1))
        self.decoder.add_module('relu2', nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.decoder.add_module('tconv1', nn.ConvTranspose1d(in_channels=64, out_channels=4, kernel_size=2, stride=1))
        self.decoder.add_module('sigmoid1', nn.Sigmoid())

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)  # Flatten the features

        # Latent representation
        x = self.latent(x)

        # Reshape for the decoder (Batch, 16, 1)
        x = x.view(-1, 16, 1)

        # Decode
        x = self.decoder(x)

        # Return None, None for mu and logvar to match VAE signature
        return x, None, None


# ==========================================
# 2. VAE-LSTM (Baseline)
# ==========================================
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = self.leaky_relu(hidden[-1])  # Take the last layer's hidden state
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, seq_len):
        hidden = self.fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.lstm(hidden)
        output = self.output_layer(output)
        return output

class VAE_LSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, latent_dim=4, output_dim=4, seq_len=4, num_layers=2):
        super(VAE_LSTM, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, output_dim, num_layers)
        self.seq_len = seq_len

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z, self.seq_len)
        return reconstructed, mu, logvar
