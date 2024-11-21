import streamlit as st
from utils import load_images, load_model, linear_interpolation, tsne_visualization, calculate_tsne

TITLE_COLOR = "#457B9D"
SUBTITLE_COLOR = "#52796f"
CAT_IMAGES = load_images()
MODEL = load_model()
IMAGES_PER_PAGE = 15
IMAGES_PER_ROW = 5
N_INTERPOLATIONS = 15


def display_image_grid(images, page_number, images_per_page, images_per_row):
    """
    Display a grid of images with pagination.

    Args:
        images (list): List of preloaded PIL Image objects.
        page_number (int): Current page number.
        images_per_page (int): Number of images to display per page.
        images_per_row (int): Number of images per row.

    Returns:
        Image: Selected Image object, or None if no image is selected.
    """
    start_idx = page_number * images_per_page
    end_idx = start_idx + images_per_page
    images_to_show = images[start_idx:end_idx]
    cols = st.columns(images_per_row, gap="small")

    for idx, img in enumerate(images_to_show):
        with cols[idx % images_per_row]:
            st.image(img, use_container_width=False, caption=f"Image {start_idx + idx + 1}", width=100)
            if st.button("Seleccionar", key=f"select_{start_idx + idx}"):
                if len(st.session_state["selected_images"]) < 2:
                    st.session_state["selected_images"].append(img)
                st.rerun()  # Trigger rerun to update the selection


def page_navigation():
    """
    Handles image gallery pagination and image selection.
    """
    if "page_number" not in st.session_state:
        st.session_state["page_number"] = 0

    total_pages = (len(CAT_IMAGES) - 1) // IMAGES_PER_PAGE
    page_number = st.session_state["page_number"]

    col1, col2 = st.columns([0.5, 0.5], gap="small")
    with col1:
        st.subheader("Galería de Imágenes")
        st.markdown(f"Mostrando página {page_number + 1} de {total_pages + 1}.")
        display_image_grid(CAT_IMAGES, page_number, IMAGES_PER_PAGE, IMAGES_PER_ROW)

        # Pagination buttons
        colA, _, colB = st.columns([0.1, 0.4, 0.1], gap="small")
        with colA:
            if st.button("Anterior", key="prev_page"):
                if st.session_state["page_number"] > 0:
                    st.session_state["page_number"] -= 1
                    st.rerun()
        with colB:
            if st.button("Siguiente", key="next_page"):
                if st.session_state["page_number"] < total_pages:
                    st.session_state["page_number"] += 1
                    st.rerun()

    with col2:
        st.subheader("Visualización del espacio latente")
        st.write(" ")

        if "tsne_results" not in st.session_state:
            tsne_results = calculate_tsne(CAT_IMAGES)
            st.session_state["tsne_results"] = tsne_results

        tsne_results = st.session_state["tsne_results"]
        selected_indexes = [
            CAT_IMAGES.index(img) for img in st.session_state["selected_images"]
        ]

        tsne_visualization(tsne_results, selected_indexes)

        if st.session_state["selected_images"]:
            st.markdown("<h3>Imágenes seleccionadas</h3>", unsafe_allow_html=True)
            st.markdown("Se han seleccionado las siguientes imágenes para la cruza:")
            sub_cols = st.columns([0.4, 0.2, 0.2, 0.4], gap="small")
            for idx, img in enumerate(st.session_state["selected_images"]):
                with sub_cols[idx+1]:
                    st.image(img, use_container_width=False, width=120, caption=f"Imagen {idx + 1}")
                    if st.button(f"Eliminar", key=f"remove_{idx}", use_container_width=True):
                        st.session_state["selected_images"].remove(img)
                        st.rerun()  # Trigger rerun to update selection


def main():
    """
    Main function to set up the app layout and logic.
    """
    st.set_page_config(page_title="Mezclador de Gatitos 🐈", layout="wide")

    st.markdown(f"<h1 style='color: {TITLE_COLOR};'>Mezclador de Gatitos 🐈</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p style="font-size: 18px;">
            Bienvenido al mezclador de gatitos, una herramienta interactiva donde puedes combinar dos adorables 
            gatos y descubrir cómo serían sus gatitos únicos. ¡Explora resultados sorprendentes y llenos de 
            ternura! ✨
        </p>
    """, unsafe_allow_html=True)

    if "selected_images" not in st.session_state:
        st.session_state["selected_images"] = []

    st.markdown(f"<h2 style='color: {SUBTITLE_COLOR};'>Seleccionar Imágenes</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p style="font-size: 16px;">
            ¡Explora nuestra galería y elige dos adorables gatitos para combinarlos y crear nuevas imágenes únicas! 
            Además, puedes visualizar las imágenes en un mapa interactivo que muestra qué tan parecidos son los gatitos 
            entre sí. Este mapa se genera a partir del espacio latente, utilizando t-SNE para reducir la dimensionalidad 
            y hacer las similitudes más fáciles de interpretar.
        </p>
    """, unsafe_allow_html=True)

    page_navigation()

    if len(st.session_state["selected_images"]) == 2:
        st.markdown(f"<h2 style='color: {SUBTITLE_COLOR};'>Gatitos Combinados</h2>", unsafe_allow_html=True)

        st.markdown("""
            <p style="font-size: 18px;">
                A continuación, se observan las imágenes generadas al combinar las características de los dos 
                gatitos seleccionados. El resultado incluye una transición visual que muestra cómo se mezclan 
                sus rasgos distintivos, desde el primero hasta el segundo, en un recorrido lleno de encanto y 
                originalidad.
            </p>
        """, unsafe_allow_html=True)


        img_1, img_2 = st.session_state["selected_images"]

        interpolated_images = linear_interpolation(MODEL, img_1, img_2, "cpu", N_INTERPOLATIONS)
        all_images = [img_1] + interpolated_images + [img_2]

        cols = st.columns(len(all_images))
        for i, interp_img in enumerate(all_images):
            with cols[i]:
                caption = "Imagen Real" if i == 0 or i == len(all_images) - 1 else f"Paso {i}"
                st.image(interp_img, use_container_width=True, caption=caption)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Realizado con ❤️ por Dylan</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
