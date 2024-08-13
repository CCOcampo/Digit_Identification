import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar el modelo
model = tf.keras.models.load_model("mnist.h5")

def recognize_digit(image):
    if image is not None:
        # Convertir la imagen a un array numpy y redimensionar
        image = np.array(image.resize((28, 28)))  # Redimensionar la imagen a (28, 28)
        image = image.reshape((1, 28, 28, 1)).astype('float32') / 255
        prediction = model.predict(image)
        return {str(i): float(prediction[0][i]) for i in range(10)}
    else:
        return 'Draw a digit'

# Configurar la interfaz de Gradio
iface = gr.Interface(
    fn=recognize_digit, 
    inputs=gr.Image(
        type='pil',  # Asegura que la imagen se recibe como un objeto PIL
        image_mode='L'  # Configura el modo de la imagen a escala de grises
    ), 
    outputs=gr.Label(num_top_classes=3),
    live=True
)

# Lanzar la interfaz
iface.launch(share=True)
