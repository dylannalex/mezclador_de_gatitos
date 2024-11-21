from math import prod

import torch.nn as nn
from torch import ones


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
