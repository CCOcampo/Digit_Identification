import os
import random
import tensorflow as tf
from PIL import Image
import shutil

#Guardar 10 ejemplos de dígitos del dataset MNIST

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (_, _) = mnist.load_data()

output_dir = 'Examples'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

random_indices = random.sample(range(len(train_images)), 10)

for i in random_indices:
    image = train_images[i]
    label = train_labels[i]
    
    image_pil = Image.fromarray(image)
    
    image_pil.save(os.path.join(output_dir, f'digit_{i}_label_{label}.png'))

print(f"Se han guardado 10 ejemplos en la carpeta '{output_dir}'.")
