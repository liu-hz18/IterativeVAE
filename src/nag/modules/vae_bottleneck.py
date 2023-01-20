
from torch import nn
from torch.nn import functional as F


class VAEBottleNeck(nn.Module):
    def __init__(self, input_size, latent_size, output_size, std_var=False, noise_level=1.):
        super(VAEBottleNeck, self).__init__()
        self.latent_size = latent_size
        self.std_var = std_var
        self.noise_level = noise_level
        self.input2latent = nn.Linear(input_size, latent_size * 2)
        self.latent2output = nn.Linear(latent_size, output_size)

    def forward(self, input):
        latent = self.input2latent(input)
        mu = latent[:, :, :self.latent_size]
        if self.std_var:
            var = latent[:, :, self.latent_size:] * 0. + 0.55
        else:
            var = latent[:, :, self.latent_size:]
        var = F.softplus(var)
        noise = mu.clone().normal_()
        z = mu + noise * (var * self.noise_level)
        output = self.latent2output(z)
        return output, z
