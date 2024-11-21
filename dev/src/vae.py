import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import Decoder
from .encoder import Encoder


class VAE(nn.Module):
    """
    A Variational Autoencoder (VAE) model with an Encoder-Decoder architecture.
    This VAE includes methods for encoding to a latent space and decoding from the
    latent space back to the original input dimensions. It also incorporates
    a reparameterization trick to introduce randomness in the latent space.

    Attributes:
        encoder (Encoder): The encoder component, which reduces the input to a latent representation.
        decoder (Decoder): The decoder component, which reconstructs the input from the latent representation.
        latent_dims (int): The number of dimensions in the latent space.
        mean_linear (nn.Linear): A linear layer to compute the mean of the latent distribution.
        log_var_linear (nn.Linear): A linear layer to compute the log variance of the latent distribution.
    """

    def __init__(
        self,
        image_h: int,
        image_w: int,
        latent_dims: int,
        hidden_channels: list[int],
        kernel_size: int,
        stride: int,
        padding: int,
        activation: nn.Module,
    ):
        """
        Initializes the VAE model by setting up the encoder, decoder, and linear
        layers for computing the mean and log variance in the latent space.

        Parameters:
            image_h (int): Height of the input image.
            image_w (int): Width of the input image.
            decoder_config (DecoderConfig): Configuration for the decoder network.
            encoder_config (EncoderConfig): Configuration for the encoder network.
            latent_dims (int): Number of dimensions in the latent space.
        """
        super(VAE, self).__init__()
        self.encoder = Encoder(
            input_height=image_h,
            input_width=image_w,
            latent_dims=latent_dims,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=activation,
        )

        self.decoder = Decoder(
            down_sampled_shape=self.encoder.down_sampled_shape,
            latent_dims=latent_dims,
            hidden_channels=hidden_channels[::-1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=activation,
        )

        self.config = {
            "image_h": image_h,
            "image_w": image_w,
            "latent_dims": latent_dims,
            "hidden_channels": hidden_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
        }

        self.init_weights()

    def encode(self, X):
        """
        Encodes the input by passing it through the encoder and then computes
        the mean and log variance for the latent distribution. A reparameterization
        trick is applied to introduce stochasticity.

        Parameters:
            X (torch.Tensor): Input tensor to be encoded.

        Returns:
            tuple: A tuple containing the latent variable (torch.Tensor), mean, and log variance.
        """
        mean, log_var = self.encoder(X)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        latent = eps * std + mean
        return latent, mean, log_var

    def decode(self, z):
        """
        Decodes the latent representation back to the original dimensions.

        Parameters:
            z (torch.Tensor): Latent variable tensor.

        Returns:
            torch.Tensor: Reconstructed output tensor.
        """
        return self.decoder(z)

    def forward(self, X):
        """
        Defines the forward pass of the VAE model. Encodes the input to a latent
        representation and then decodes it back to reconstruct the original input.

        Parameters:
            X (torch.Tensor): Input tensor to the VAE model.

        Returns:
            tuple: A tuple containing the reconstructed output, mean, and log variance.
        """
        z, mean, log_var = self.encode(X)
        return self.decode(z), mean, log_var

    def init_weights(self):
        """
        Initializes the weights of the VAE model using the Kaiming initialization method.
        """
        for module in self.modules():
            if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                continue

            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)


class BetaLoss(nn.Module):
    def __init__(self, beta: float):
        super(BetaLoss, self).__init__()
        self.beta = beta

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        media: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
        latent_loss = -0.5 * torch.sum(1 + log_var - log_var.exp() - media.pow(2))
        return reconstruction_loss + self.beta * latent_loss
