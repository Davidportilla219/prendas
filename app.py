from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
	page_title="Predictor de prendas",
	page_icon="👕",
	layout="centered",
)

CLASS_NAMES = [
	"Camiseta",
	"Pantalón",
	"Suéter",
	"Vestido",
	"Abrigo",
	"Zapato de playa",
	"Camisa",
	"Zapatilla",
	"Bolso",
	"Botines",
]
MODEL_PATH = Path(__file__).with_name("prendas.keras")


@st.cache_resource
def load_model():
	return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(image: Image.Image) -> np.ndarray:
	grayscale = ImageOps.grayscale(image)
	resized = grayscale.resize((28, 28), Image.Resampling.LANCZOS)
	return np.asarray(resized, dtype=np.float32) / 255.0


def predict(image: Image.Image) -> tuple[str, float, np.ndarray]:
	model = load_model()
	pixels = prepare_image(image)
	probabilities = model.predict(pixels[np.newaxis, ...], verbose=0)[0]
	predicted_index = int(np.argmax(probabilities))
	return (
		CLASS_NAMES[predicted_index],
		float(probabilities[predicted_index]),
		probabilities,
	)


def show_prediction(image: Image.Image) -> None:
	predicted_class, confidence, probabilities = predict(image)
	st.subheader(f"Predicción: {predicted_class}")
	st.metric("Confianza", f"{confidence:.1%}")
	st.bar_chart(
		{name: float(value) for name, value in zip(CLASS_NAMES, probabilities)},
		horizontal=True,
	)


st.title("Predictor de prendas con TensorFlow")
st.write("Dibuja una prenda o sube una imagen para que el modelo la clasifique.")

draw_tab, upload_tab = st.tabs(["Dibujar", "Subir imagen"])

with draw_tab:
	st.subheader("Dibuja sobre el fondo negro")
	canvas_result = st_canvas(
		fill_color="rgba(255, 255, 255, 1)",
		stroke_width=12,
		stroke_color="#FFFFFF",
		background_color="#000000",
		height=280,
		width=280,
		drawing_mode="freedraw",
		key="clothing_canvas",
	)
	if st.button("Predecir dibujo", type="primary", use_container_width=True):
		if canvas_result.image_data is None:
			st.warning("Dibuja una imagen antes de hacer la predicción.")
		else:
			drawing = Image.fromarray(
				np.uint8(canvas_result.image_data[:, :, :3]), mode="RGB"
			)
			show_prediction(drawing)

with upload_tab:
	st.subheader("Carga una imagen")
	uploaded_file = st.file_uploader(
		"Selecciona una imagen",
		type=["png", "jpg", "jpeg", "webp"],
	)
	if uploaded_file is not None:
		uploaded_image = Image.open(uploaded_file).convert("RGB")
		st.image(uploaded_image, caption="Imagen cargada", width="stretch")
		st.caption("La imagen se convertirá a escala de grises y a 28 x 28 píxeles.")
		if st.button("Predecir imagen", type="primary", use_container_width=True):
			show_prediction(uploaded_image)

st.divider()
st.subheader("Instrucciones")
st.markdown(
	"""
	- Para dibujar, representa la prenda con trazos blancos sobre el fondo negro.
	- Las imágenes se convierten a escala de grises, se ajustan a 28 x 28 píxeles
	  y se normalizan dividiendo sus valores entre 255.
	- Para obtener mejores resultados, carga o dibuja imágenes similares a las
	  usadas durante el entrenamiento del modelo.
	- El modelo devuelve probabilidades mediante una capa de salida `softmax`;
	  se muestra como predicción la clase con mayor probabilidad.
	"""
)
