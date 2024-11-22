import os
from math import prod

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import ones
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
import pandas as pd
import streamlit as st
import altair as alt

# /mount/src/mezclador_de_gatitos/prod/app.py

class UpSample(nn.Module):
    """
    A custom up-sampling layer that increases the spatial dimensions of the input
    using a transposed convolution, followed by batch normalization and activation.

    Attributes:
        conv_trans (nn.ConvTranspose2d): A 2D transposed convolution layer.
        batch_norm (nn.BatchNorm2d): Batch normalization layer applied after the transposed convolution.
        activation (nn.LeakyReLU): LeakyReLU activation function applied after batch normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        activation: nn.Module,
    ):
        """
        Initializes the UpSample layer with transposed convolution, batch normalization,
        and activation.

        Parameters:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the transposed convolution.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding added to both sides of the input.
        """
        super(UpSample, self).__init__()
        self.conv_trans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.activation = activation

    def forward(self, X):
        """
        Defines the computation performed at every forward pass.

        Parameters:
            X (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with upsampled spatial dimensions and `out_channels` channels.
        """
        X = self.conv_trans(X)
        X = self.activation(X)
        return X


class Decoder(nn.Module):
    """
    A Decoder network composed of multiple UpSample layers for progressive up-sampling
    and a final transposed convolution to produce an output with three channels.

    Attributes:
        decoder (nn.Sequential): A sequence of UpSample layers followed by a final transposed
            convolution to obtain the desired output dimensions.
    """

    def __init__(
        self,
        down_sampled_shape: tuple[int, int, int],
        latent_dims: int,
        hidden_channels: list[int],
        kernel_size: int,
        stride: int,
        padding: int,
        activation: nn.Module,
    ):
        """
        Initializes the Decoder with a stack of UpSample layers.

        Parameters:
            down_sampled_shape (tuple[int, int, int]): Shape of the down-sampled tensor from the encoder.
            hidden_channels (list[int]): A list where each element represents the number
                of channels in each layer of the decoder.
            kernel_size (int): Kernel size for each transposed convolution layer.
            stride (int): Stride for each transposed convolution layer.
            padding (int): Padding for each transposed convolution layer.
        """
        super(Decoder, self).__init__()
        layers = []
        for i in range(len(hidden_channels) - 1):
            layers.append(
                UpSample(
                    in_channels=hidden_channels[i],
                    out_channels=hidden_channels[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=activation,
                )
            )

        # Final layer to map to output channels
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, prod(down_sampled_shape)),
            nn.Unflatten(1, down_sampled_shape),
            *layers,
            nn.Sigmoid(),
        )

    def forward(self, X):
        """
        Defines the computation performed at every forward pass.

        Parameters:
            X (torch.Tensor): Input tensor to the Decoder with shape (batch_size, hidden_channels[0], height, width).

        Returns:
            torch.Tensor: Decoded output tensor with shape (batch_size, 3, output_height, output_width).
        """
        return self.decoder(X)


class DownSample(nn.Module):
    """
    A custom down-sampling layer that reduces the spatial dimensions of the input
    using a convolutional layer, followed by batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): A 2D convolution layer for down-sampling.
        batch_norm (nn.BatchNorm2d): Batch normalization layer applied after convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        activation: nn.Module,
    ):
        """
        Initializes the DownSample layer with convolution, batch normalization,
        and activation.

        Parameters:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding added to both sides of the input.
        """
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation

    def forward(self, X):
        """
        Defines the computation performed at every forward pass.

        Parameters:
            X (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with downsampled spatial dimensions and `out_channels` channels.
        """
        X = self.conv(X)
        X = self.activation(X)
        return X


class Encoder(nn.Module):
    """
    An Encoder network composed of multiple DownSample layers for progressive down-sampling,
    followed by a max pooling, flattening, and a linear layer to reduce to latent dimensions.

    Attributes:
        encoder (nn.Sequential): A sequence of DownSample layers, a max pooling layer, a flattening layer,
            and a final linear layer to obtain the latent representation.
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        latent_dims: int,
        hidden_channels: list[int],
        kernel_size: int,
        stride: int,
        padding: int,
        activation: nn.Module,
    ):
        """
        Initializes the Encoder with a stack of DownSample layers and a final linear layer for
        producing the latent representation.

        Parameters:
            input_height (int): Height of the input image.
            input_width (int): Width of the input image.
            latent_dims (int): Number of dimensions in the latent space representation.
            hidden_channels (list[int]): A list where each element represents the number of channels
                in each layer of the encoder.
            kernel_size (int): Kernel size for each convolutional layer.
            stride (int): Stride for each convolutional layer.
            padding (int): Padding for each convolutional layer.
            activation (nn.Module): Activation function applied after each convolutional layer.

        Attributes:
            down_sampled_shape (tuple): Shape of the tensor after passing through the DownSample layers.
        """
        super(Encoder, self).__init__()
        layers = []
        for i in range(len(hidden_channels) - 1):
            layers.append(
                DownSample(
                    in_channels=hidden_channels[i],
                    out_channels=hidden_channels[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=activation,
                )
            )
        self.convs = nn.Sequential(*layers)
        self.down_sampled_shape = self.convs(
            ones(1, hidden_channels[0], input_height, input_width)
        ).shape[1:]

        self._mean_linear = nn.Linear(
            in_features=prod(self.down_sampled_shape),
            out_features=latent_dims,
        )

        self._log_var_linear = nn.Linear(
            in_features=prod(self.down_sampled_shape),
            out_features=latent_dims,
        )

    def forward(self, X):
        """
        Defines the computation performed at every forward pass.

        Parameters:
            X (torch.Tensor): Input tensor to the Encoder with shape (batch_size, hidden_channels[0], height, width).

        Returns:
            torch.Tensor: Encoded tensor with shape (batch_size, latent_dims).
        """
        X = self.convs(X)
        X = nn.Flatten()(X)
        mean = self._mean_linear(X)
        log_var = self._log_var_linear(X)
        return mean, log_var


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


# Cargar imagen desde la ruta
def load_image(image_path):
    """Cargar y redimensionar una imagen desde un archivo."""
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Redimensionar a 64x64
    return img


# Cargar todas las imágenes de gatos desde el directorio 'data/cats'
def load_images():
    """Cargar rutas de imágenes de gatos."""
    cat_images = os.listdir(os.path.join("prod", "data", "cats"))
    cat_images = [
        img for img in cat_images if img.endswith(".jpg") or img.endswith(".jpeg")
    ]
    cat_images = sorted(cat_images, key=lambda name: int(name.split(".")[0]))
    return [load_image(os.path.join("prod", "data", "cats", img)) for img in cat_images]


# Cargar el modelo VAE preentrenado
def load_model():
    vae_config = {
        "image_h": 64,
        "image_w": 64,
        "latent_dims": 500,
        "hidden_channels": [3, 32, 64, 128, 256],
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "activation": torch.nn.LeakyReLU(),
    }
    vae = VAE(**vae_config).to("cpu")
    vae.load_state_dict(
        torch.load(os.path.join("prod", "models", "vae.pt"), map_location=torch.device("cpu"))
    )
    return vae


# Realizar interpolación lineal entre dos imágenes en el espacio latente
def linear_interpolation(
    model: VAE, img_1: Image.Image, img_2: Image.Image, device: str, steps: int = 10
):
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    img_1_tensor = transform(img_1).unsqueeze(0).to(device)
    img_2_tensor = transform(img_2).unsqueeze(0).to(device)

    with torch.no_grad():
        latent_1, _, _ = model.encode(img_1_tensor)
        latent_2, _, _ = model.encode(img_2_tensor)

        v1 = latent_1.to("cpu")
        v2 = latent_2.to("cpu")

        interps = np.array(
            [np.linspace(v1[i], v2[i], steps + 2) for i in range(v1.shape[0])]
        ).T
        interps = torch.tensor(interps).to(device).squeeze()
        interps = interps.permute(1, 0)
        decoded_interps = model.decode(interps.to(device))

    decoded_interps = [
        img.squeeze().permute(1, 2, 0).cpu().numpy() for img in decoded_interps
    ]
    return decoded_interps[1:-1]  # Excluir los primeros y últimos pasos


def calculate_tsne(images):
    """Calcula el t-SNE para un conjunto de imágenes.

    Args:
        images: Lista de imágenes de entrada.

    Returns:
        tsne_results: Resultados de t-SNE proyectados en 2D.
    """
    # Preprocesar las imágenes
    image_vectors = [np.array(img).flatten() for img in images]  # Aplana las imágenes
    image_vectors = np.array(image_vectors)

    # Aplicar t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(image_vectors)

    return tsne_results


def tsne_visualization(tsne_results, selected_indexes):
    """
    Visualize t-SNE results using Streamlit's Altair scatter plot.

    Args:
        tsne_results (ndarray): The t-SNE results, a 2D array with coordinates.
        selected_indexes (list): A list of indexes for selected images to highlight.

    Returns:
        None: Displays the scatter plot with Streamlit.
    """
    # Convert t-SNE results to a DataFrame
    tsne_df = pd.DataFrame(tsne_results, columns=["x", "y"])

    # Default settings for all points
    tsne_df["color"] = "#f4a261"  # Default color
    tsne_df["size"] = 50  # Default size

    # Update settings for selected points, if any
    if selected_indexes:
        tsne_df.loc[selected_indexes, "color"] = "#2a9d8f"  # Highlighted color
        tsne_df.loc[selected_indexes, "size"] = 100  # Highlighted size

    # Create an Altair scatter plot
    scatter_chart = (
        alt.Chart(tsne_df)
        .mark_circle()
        .encode(
            x=alt.X("x", title="Dimensión t-SNE 1"),
            y=alt.Y("y", title="Dimensión t-SNE 2"),
            color=alt.Color("color:N", scale=None, legend=None),
            size=alt.Size("size:Q", legend=None),
        )
    )
    st.altair_chart(scatter_chart, use_container_width=True)
