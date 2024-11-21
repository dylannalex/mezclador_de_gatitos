import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
import pandas as pd
import streamlit as st
import altair as alt


os.sys.path.append("../dev")
from src import VAE


# Cargar imagen desde la ruta
def load_image(image_path):
    """Cargar y redimensionar una imagen desde un archivo."""
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Redimensionar a 64x64
    return img


# Cargar todas las imágenes de gatos desde el directorio 'data/cats'
def load_images():
    """Cargar rutas de imágenes de gatos."""
    cat_images = os.listdir(os.path.join("data", "cats"))
    cat_images = [img for img in cat_images if img.endswith(".jpg") or img.endswith(".jpeg")]
    cat_images = sorted(cat_images, key=lambda name: int(name.split(".")[0]))
    return [load_image(os.path.join("data", "cats", img)) for img in cat_images]


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
        torch.load(os.path.join("models", "vae.pt"), map_location=torch.device("cpu"))
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
            x=alt.X("x", title="t-SNE Dimension 1"),
            y=alt.Y("y", title="t-SNE Dimension 2"),
            color=alt.Color("color:N", scale=None, legend=None),
            size=alt.Size("size:Q", legend=None),
        )
    )
    st.altair_chart(scatter_chart, use_container_width=True)
