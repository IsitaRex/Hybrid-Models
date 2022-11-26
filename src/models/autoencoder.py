import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Autoencoder, self).__init__()
        self.encoder_hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True))

        self.encoder_output = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU(True))


        self.decoder_hidden = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(True))
        
        self.decoder_output = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder_hidden(x)
        x = self.encoder_output(x)
        x = self.decoder_hidden(x)
        x = self.decoder_output(x)
        return x