# Mezclador de Gatitos 🐱

[![Página Web](https://img.shields.io/badge/P%C3%A1gina_Web-Mezclador%20de%20Gatitos-blue)](https://mezclador-gatitos.streamlit.app/)
[![licence](https://img.shields.io/github/license/dylannalex/mezclador_de_gatitos?color=blue)](https://github.com/dylannalex/mezclador_de_gatitos/blob/main/LICENSE)

¿Alguna vez has soñado con poder fusionar dos adorables gatitos en uno solo? ¡Pues ahora puedes hacerlo realidad con el **Mezclador de Gatitos**! Este proyecto se basa en la inteligencia artificial y el poder de las redes neuronales para combinar imágenes de gatitos, creando nuevas combinaciones visuales que no solo son asombrosas, sino también irresistiblemente tiernas.

<p align="center"> <img src="../media/interpolation_example.png?raw=true" /> </p>

Con el Mezclador de Gatitos puedes:

- ✨ **Combinar imágenes de gatitos:** Selecciona dos gatitos y descubre fascinantes fusiones llenas de ternura.

- ✨ **Descubrir el mapa oculto de los gatitos:** Explora cómo la red neuronal organiza y representa a los gatitos en un espacio bidimensional.

**¿Listo para descubrir combinaciones que nunca imaginaste?** 👉 [Explora ahora y crea tus propios gatitos únicos](https://mezclador-gatitos.streamlit.app/).

> El proyecto fue desarrollado como parte de la materia Redes Neuronales Profundas en la UTN FRM, bajo la guía del profesor Ing. Pablo Marinozi. Este trabajo es un testimonio de creatividad, aprendizaje y, por supuesto, ¡mucha ternura! ❤️

## 📚 Tabla de Contenidos

1. [Organización del Repositorio](#-organización-del-repositorio)
2. [Paquete para la implementación del VAE](#-paquete-para-la-implementación-del-vae)
3. [Instalación y Ejecución](#-instalación-y-ejecución)

## 🐈 Organización del Repositorio

El repositorio sigue una estructura organizada para facilitar el desarrollo, entrenamiento del modelo y ejecución de la aplicación:

### 1. `data/`
Contiene los datasets utilizados para el entrenamiento y evaluación del modelo:

- **cats.zip**: Dataset con imágenes de gatitos.
- **data_exploration.ipynb**: Notebook de Jupyter con análisis y exploración inicial de los datos.

### 2. `dev/`
Incluye notebooks y código relacionado con el desarrollo experimental del modelo VAE:

- **src/**: Código fuente del Autoencoder Variacional (VAE).
- **model_training.ipynb**: Notebook donde se entrena el modelo utilizando el dataset de gatitos.

### 3. `prod/`
Carpeta destinada a la implementación de la aplicación final:

- **app.py**: Archivo principal que ejecuta la aplicación web interactiva con Streamlit.
- **models/vae.pt**: Archivo con los pesos del modelo VAE entrenado en formato PyTorch.
- **requirements.txt**: Lista de dependencias necesarias para ejecutar el proyecto.
- **utils.py**: Funciones auxiliares para el preprocesamiento y carga del modelo.


## 🐈 Paquete para la implementación del VAE

Para facilitar el entrenamiento de los modelos, el paquete `dev/src/` contiene tres funcionalidades clave:

### 1. VAE (Variational Autoencoder)

El modelo VAE implementa un codificador (encoder) y un decodificador (decoder) para comprimir y reconstruir datos. Este modelo utiliza el truco de reparametrización para aprender una distribución latente que captura la estructura subyacente de los datos.

#### Principales parámetros del constructor
- **image_h** (int): Altura de la imagen de entrada.
- **image_w** (int): Ancho de la imagen de entrada.
- **latent_dims** (int): Dimensionalidad del espacio latente.
- **hidden_channels** (list[int]): Lista con los canales ocultos de cada capa convolucional.
- **kernel_size** (int): Tamaño del kernel para las convoluciones.
- **stride** (int): Stride (desplazamiento) utilizado en las convoluciones.
- **padding** (int): Padding aplicado en las convoluciones.
- **activation** (`nn.Module`): Función de activación utilizada en las capas.

#### Métodos principales
- **encode(X)**: Codifica una entrada `X` y devuelve la representación latente, junto con la media y la varianza logarítmica.
- **decode(z)**: Decodifica una representación latente `z` para reconstruir la entrada.
- **forward(X)**: Encadena las fases de codificación y decodificación en un único paso.
- **init_weights()**: Inicializa los pesos del modelo utilizando el método de inicialización de Kaiming.

### 2. BetaLoss

Esta clase implementa una pérdida personalizada basada en la combinación de la pérdida de reconstrucción y la pérdida latente ponderada por un factor $\beta$.

#### Principales parámetros del constructor
- **beta** (float): Factor de ponderación para la pérdida latente.

#### Método principal:
- **forward(x, x_hat, mean, log_var)**
  - **x**: Entrada original.
  - **x_hat**: Salida reconstruida por el VAE.
  - **mean**: Media de la distribución latente.
  - **log_var**: Varianza logarítmica de la distribución latente.
  - **Retorna**: La suma ponderada de la pérdida de reconstrucción y la pérdida latente.

### 3. Función de entrenamiento

La función `train()` entrena el modelo por un número especificado de épocas, guarda checkpoints periódicamente y calcula las métricas de validación.

#### Principales parámetros
- **model** (`nn.Module`): Modelo a entrenar.
- **train_loader** (`DataLoader`): DataLoader para los datos de entrenamiento.
- **val_loader** (`DataLoader` o `None`): DataLoader para los datos de validación (opcional).
- **epochs** (int): Número total de épocas para entrenar.
- **epochs_per_checkpoint** (int): Intervalo de épocas para guardar checkpoints.
- **device** (str): Dispositivo donde se ejecutará el entrenamiento (e.g., `cpu` o `cuda`).
- **model_dir** (str): Directorio donde se guardarán los pesos y el historial.
- **optimizer** (`torch.optim.Optimizer`): Optimizador utilizado para actualizar los pesos.
- **loss_function** (`nn.Module`): Función de pérdida utilizada en el entrenamiento.

#### Flujo principal
1. **Preparación del modelo**:
   - Verifica si existen pesos previamente entrenados en el directorio especificado.
   - Carga los pesos más recientes si están disponibles.
2. **Entrenamiento**:
   - Por cada época, entrena el modelo utilizando el `train_loader` y calcula el promedio de la pérdida.
   - Si se especifica un `val_loader`, calcula la pérdida promedio en el conjunto de validación.
3. **Checkpointing**:
   - Guarda los pesos del modelo y actualiza las métricas en un archivo `.csv` después de cada época.

Esta estructura modular permite entrenar un modelo VAE de manera eficiente y realizar un seguimiento detallado del proceso.


## 🐈 Instalación y Ejecución

Para ejecutar el proyecto en tu máquina local, sigue estos pasos:

### 1. Clonar el repositorio

```bash
$ git clone https://github.com/dylannalex/mezclador_de_gatitos.git
```

### 2. Instalar dependencias
```bash
$ pip install -r prod/requirements.txt
```

### 3. Ejecutar la aplicación
```bash
$ streamlit run prod/app.py
```

