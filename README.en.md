# Kitten Mixer 🐱

[![Website](https://img.shields.io/badge/Website-Kitten%20Mixer-blue)](https://mezclador-gatitos.streamlit.app/)
[![licence](https://img.shields.io/github/license/dylannalex/mezclador_de_gatitos?color=blue)](https://github.com/dylannalex/mezclador_de_gatitos/blob/main/LICENSE)
[![English](https://img.shields.io/badge/🌐-%20English-blue)](https://github.com/dylannalex/mezclador_de_gatitos/blob/main/README.en.md)
[![Español](https://img.shields.io/badge/🌐-%20Español-blue)](https://github.com/dylannalex/mezclador_de_gatitos/blob/main/README.md)


Have you ever dreamed of merging two adorable kittens into one? Now you can make it a reality with the **Kitten Mixer**! This project is based on artificial intelligence and the power of neural networks to combine kitten images, creating new visual combinations that are not only amazing but also irresistibly cute.

<p align="center"> <img src="../media/interpolation_example.png?raw=true" /> </p>

With the Kitten Mixer you can:

- ✨ **Combine kitten images:** Select two kittens and discover fascinating fusions full of cuteness.

- ✨ **Discover the hidden map of kittens:** Explore how the neural network organizes and represents kittens in a two-dimensional space.

**Ready to discover combinations you never imagined?** 👉 [Explore now and create your own unique kittens](https://mezclador-gatitos.streamlit.app/).

> The project was developed as part of the Deep Neural Networks course at UTN FRM, under the guidance of Professor Ing. Pablo Marinozi. This work is a testament to creativity, learning, and of course, a lot of cuteness! ❤️

## 📚 Table of Contents

1. [Repository Structure](#-repository-structure)
2. [Package for VAE Implementation](#-package-for-vae-implementation)
3. [Installation and Execution](#-installation-and-execution)

## 🐈 Repository Structure

The repository follows an organized structure to facilitate development, model training, and application execution:

### 1. `data/`
Contains the datasets used for training and evaluating the model:

- **cats.zip**: Dataset with kitten images.
- **data_exploration.ipynb**: Jupyter notebook with initial data analysis and exploration.

### 2. `dev/`
Includes notebooks and code related to the experimental development of the VAE model:

- **src/**: Source code for the Variational Autoencoder (VAE).
- **model_training.ipynb**: Notebook where the model is trained using the kitten dataset.

### 3. `prod/`
Folder dedicated to the implementation of the final application:

- **app.py**: Main file that runs the interactive web application with Streamlit.
- **models/vae.pt**: File with the trained VAE model weights in PyTorch format.
- **requirements.txt**: List of dependencies required to run the project.
- **utils.py**: Helper functions for preprocessing and loading the model.


## 🐈 Package for VAE Implementation

To facilitate model training, the `dev/src/` package contains three key functionalities:

### 1. VAE (Variational Autoencoder)

The VAE model implements an encoder and decoder to compress and reconstruct data. This model uses the reparameterization trick to learn a latent distribution that captures the underlying structure of the data.

#### Main constructor parameters
- **image_h** (int): Height of the input image.
- **image_w** (int): Width of the input image.
- **latent_dims** (int): Dimensionality of the latent space.
- **hidden_channels** (list[int]): List of hidden channels for each convolutional layer.
- **kernel_size** (int): Kernel size for convolutions.
- **stride** (int): Stride used in convolutions.
- **padding** (int): Padding applied in convolutions.
- **activation** (`nn.Module`): Activation function used in layers.

#### Main methods
- **encode(X)**: Encodes an input `X` and returns the latent representation, along with the mean and log variance.
- **decode(z)**: Decodes a latent representation `z` to reconstruct the input.
- **forward(X)**: Chains the encoding and decoding phases in one step.
- **init_weights()**: Initializes model weights using the Kaiming initialization method.

### 2. BetaLoss

This class implements a custom loss based on a combination of reconstruction loss and latent loss weighted by a $\beta$ factor.

#### Main constructor parameters
- **beta** (float): Weighting factor for the latent loss.

#### Main method:
- **forward(x, x_hat, mean, log_var)**
  - **x**: Original input.
  - **x_hat**: Output reconstructed by the VAE.
  - **mean**: Mean of the latent distribution.
  - **log_var**: Log variance of the latent distribution.
  - **Returns**: The weighted sum of the reconstruction loss and the latent loss.

### 3. Training Function

The `train()` function trains the model for a specified number of epochs, saves periodic checkpoints, and calculates validation metrics.

#### Main parameters
- **model** (`nn.Module`): Model to train.
- **train_loader** (`DataLoader`): DataLoader for training data.
- **val_loader** (`DataLoader` or `None`): DataLoader for validation data (optional).
- **epochs** (int): Total number of epochs to train.
- **epochs_per_checkpoint** (int): Epoch interval to save checkpoints.
- **device** (str): Device where the training will run (e.g., `cpu` or `cuda`).
- **model_dir** (str): Directory where model weights and history will be saved.
- **optimizer** (`torch.optim.Optimizer`): Optimizer used to update the weights.
- **loss_function** (`nn.Module`): Loss function used during training.

#### Main flow
1. **Model preparation**:
   - Checks if pre-trained weights exist in the specified directory.
   - Loads the most recent weights if available.
2. **Training**:
   - For each epoch, trains the model using the `train_loader` and calculates the average loss.
   - If a `val_loader` is provided, calculates the average loss on the validation set.
3. **Checkpointing**:
   - Saves model weights and updates metrics in a `.csv` file after each epoch.

This modular structure allows for efficient VAE model training and detailed tracking of the process.


## 🐈 Installation and Execution

To run the project on your local machine, follow these steps:

### 1. Clone the repository

```bash
$ git clone https://github.com/dylannalex/mezclador_de_gatitos.git
```

### 2. Install dependencies

```bash
$ pip install -r prod/requirements.txt
```

### 3. Run the application
```bash
$ streamlit run prod/app.py
```
