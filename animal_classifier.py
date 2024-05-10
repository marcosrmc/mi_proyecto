import os
import cv2
import numpy as np
import tensorflow as tf

class AnimalClassifier:
    def __init__(self):
        # Cargar tu propio modelo preentrenado para la detección y clasificación
        self.model_path = 'mimodelo.tflite'
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Obtener nombres de clases desde el directorio 'animales/train/'
        self.class_names = self.obtener_nombres_clases()

    def obtener_nombres_clases(self):
        # Obtener nombres de clases desde el directorio 'animales/train/'
        class_names = os.listdir('animales/train/')
        class_names.sort()  # Ordenar para consistencia
        return class_names

    def predict(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Lee la imagen en color
        img_resized = cv2.resize(img, (128, 128))  # Redimensiona la imagen a 128x128
        img_resized = img_resized.astype(np.float32) / 255.0  # Normaliza los valores de píxeles entre 0 y 1
        img_resized = np.expand_dims(img_resized, axis=0)

        # Establecer los valores del tensor de entrada
        self.interpreter.set_tensor(self.input_details[0]['index'], img_resized)
        self.interpreter.invoke()

        # Obtener el tensor de salida
        result = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Obtener la clase con mayor probabilidad
        predicted_index = np.argmax(result)
        predicted_class = self.class_names[predicted_index]

        return predicted_class, result[predicted_index]  # Devuelve la clase y su probabilidad
