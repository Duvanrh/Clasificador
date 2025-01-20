import streamlit as st
from tensorflow.keras.models import load_model  # O ajusta según el framework
from PIL import Image
import numpy as np

# Configura la página
st.set_page_config(page_title="Clasificación de Imágenes", layout="centered")

# Carga el modelo (usa @st.cache_resource para evitar recargas constantes)
@st.cache_resource
def cargar_modelo():
    ruta_modelo = "./Modelo/clasi_animales.keras"
    return load_model(ruta_modelo)

modelo = cargar_modelo()

# Clase para interpretar las predicciones (ajusta según tu modelo)
CLASES = ['Aranha', 'Ardilla', 'Caballo', 'Elefante', 'Gallina', 'Gato', 'Mariposa', 'Oveja', 'Perro', 'Vaca']  # Cambia según tus etiquetas

# Título de la aplicación
st.title("Clasificador de Animales con IA")
st.markdown(
    "Sube una imagen y el modelo clasificará la categoría a la que pertenece. "
    "Este modelo está entrenado para distinguir entre las siguientes clases:"
)
st.write(", ".join(CLASES))

# Carga de la imagen
archivo_imagen = st.file_uploader("Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])

if archivo_imagen:
    # Mostrar la imagen cargada
    imagen = Image.open(archivo_imagen)
    st.image(imagen, caption="Imagen cargada", use_column_width=True)
    
    # Procesar la imagen para el modelo
    st.write("Procesando la imagen...")
    imagen_resized = imagen.resize((224, 224))  # Cambia el tamaño según lo que espera tu modelo
    #imagen_array = np.array(imagen_resized) / 255.0  # Normaliza si es necesario
    imagen_array = np.expand_dims(imagen_resized, axis=0)  # Añade dimensión batch

    # Realizar la predicción
    prediccion = modelo.predict(imagen_array)
    clase_predicha = CLASES[np.argmax(prediccion)]
    confianza = np.max(prediccion) * 100

    # Mostrar resultados
    st.success(f"Predicción: **{clase_predicha}**")
    st.info(f"Confianza: **{confianza:.2f}%**")

# Información adicional o descarga de resultados
st.markdown("---")
st.write("Desarrollado con Streamlit y redes neuronales.")