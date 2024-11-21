from math import prod

import torch.nn as nn


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
