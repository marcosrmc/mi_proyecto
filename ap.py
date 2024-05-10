from flask import Flask, render_template, request, session
import os
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__, static_folder="static")
app.secret_key = 'your_secret_key'  
app.config['SESSION_COOKIE_SECURE'] = True  
app.config['UPLOAD_FOLDER'] = 'static' 

model_path = 'mimodelo.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = os.listdir('animales/train/')
class_names.sort()  

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('predecir.html', error="No se proporcionó ninguna imagen.")

        image_file = request.files['image']
        if image_file.filename == '':
            return render_template('predecir.html', error="El archivo de imagen no tiene un nombre válido.")

        # Establecer un nombre fijo para la imagen
        image_filename = 'temp_image.jpg'
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

        # Guardar el archivo de imagen
        image_file.save(image_path)

        predicted_class, confidence = predict(image_path)

        return render_template('predecir.html', image_path=image_filename, predicted_class=predicted_class, confidence=confidence)

    if 'image_path' in session:
        image_path = session['image_path']
    else:
        image_path = None

    return render_template('index.html', image_path=image_path)

@app.route('/predecir')
def predecir():
    return render_template('predecir.html')

def predict(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  
    img_resized = cv2.resize(img, (128, 128))  
    img_resized = img_resized.astype(np.float32) / 255.0  
    img_resized = np.expand_dims(img_resized, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_resized)
    interpreter.invoke()

    result = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_index = np.argmax(result)
    predicted_class = class_names[predicted_index]

    return predicted_class, result[predicted_index]  

if __name__ == '__main__':
    app.run(debug=True)
