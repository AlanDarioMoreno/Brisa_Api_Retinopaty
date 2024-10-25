from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


app = Flask(__name__)


try:
    model = tf.keras.models.load_model('./model/Modelo002-68182.keras')

    if isinstance(model, tf.keras.Model):
        model.summary()
    else:
        print("El objeto cargado no es un modelo de Keras.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")


 #Aplico Gaussian Blur a la imagen
def add_gaussian_blur(image, sigmaX=10):   
    blurred_image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return blurred_image


@app.route('/')
def home():
    return "Brisa API Home"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se recibió un archivo."}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No se seleccionó un archivo."}), 400

        # Abre la imagen y la convierte a RGB
        img = Image.open(file.stream).convert('RGB')

        # Convertir la imagen a un array de numpy (esperado por OpenCV)
        img_array = np.array(img)

        # Aplicar el filtro GaussianBlur
        blurred_image = add_gaussian_blur(img_array)

        # Redimensionar la imagen a 224x224
        blurred_image = cv2.resize(blurred_image, (224, 224))

        # Normalizar la imagen (valores entre 0 y 1)
        blurred_image = blurred_image.astype(np.float32) / 255.0

        # Agregar dimensión del batch
        blurred_image = np.expand_dims(blurred_image, axis=0)

        # Realiza la predicción
        prediction = model.predict(blurred_image)
        result = prediction[0].tolist()

        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)