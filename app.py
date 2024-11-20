from flask import Flask, request, jsonify
from torchvision import transforms
import torch
import cv2
from PIL import Image
from dotenv import load_dotenv
import numpy as np
import os
from Model import Model
from modelInicialization import modelInicialization
import warnings

#Ignora un warning de seguridad de Pytorch
warnings.filterwarnings("ignore", category=FutureWarning) 

IMAGE_SIZE = 512
RETINO_CLASS= 5
RETINO_NORETINO= 2
RETINO_FILTRO= 3

def get_retinopathy_description(clase):
    switch = {
        0: ["TIENE RETINOPATÍA", "LEVE"],
        1: ["TIENE RETINOPATÍA", "MODERADA"],
        2: ["NO TIENE RETINOPATÍA", "NO_DR"],
        3: ["TIENE RETINOPATÍA", "PROLIFERADA"],
        4: ["TIENE RETINOPATÍA", "SEVERA"]
    }
    return switch.get(clase, ["Clase no válida", "Descripción no válida"])

app = Flask(__name__)
load_dotenv()

#Se instancian los modelos de clasificacion y retina/No Retina
modelRetinoNoRetino= modelInicialization(os.getenv("model_name_RE_NORE"),os.getenv("local_model_path_retina_NoRetina"),RETINO_NORETINO)
modelMild= modelInicialization(os.getenv("model_name_filtroGaussian"),os.getenv("local_model_path_filtroGaussian"),RETINO_FILTRO)
modelClasification = modelInicialization(os.getenv("model_name"),os.getenv("local_model_path"),RETINO_CLASS)

# Definir la transformación personalizada para aplicar Gaussian Blur
class GaussianBlurTransform:
    def __init__(self, sigmaX=10):
        self.sigmaX = sigmaX

    def __call__(self, image):
        # Convertir la imagen a un array de NumPy
        image_np = np.array(image)
        # Aplicar el filtro Gaussian Blur en OpenCV
        blurred_image = cv2.addWeighted(image_np, 4, cv2.GaussianBlur(image_np, (0, 0), self.sigmaX), -4, 128)
        # Convertir la imagen de nuevo a formato PIL
        blurred_image = Image.fromarray(blurred_image)
        return blurred_image

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizacion de color RGB y desviacion del modelo RestNet50
])

preprocess_gaussian = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    GaussianBlurTransform(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizacion de color RGB y desviacion del modelo RestNet50
])

@app.route('/', methods=['GET'])
def home():
    return "Hola, soy la home."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se recibió un archivo."}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No se seleccionó un archivo."}), 400

        img = Image.open(file.stream).convert('RGB')
       
        processed_img = preprocess(img).unsqueeze(0)  # Transformaciones y batch dimension

        # 4. Realizar la predicción
         # Predicción con el modelo 1
        with torch.no_grad():
            output1 = modelClasification(processed_img)
            probabilities1 = torch.nn.functional.softmax(output1, dim=1)
            confidence1, pred1 = torch.max(probabilities1, 1)

        print(f"Modelo 1: Predicción {pred1.item()}, Confianza {confidence1.item():.2f}")
        print(output1)

        # Si la predicción del modelo 1 es clase 2 con confianza menor a 98%, usar modelo 2
        if pred1.item() == 2 and confidence1.item() < 0.98:
            print("Evaluando con modelo 2 debido a baja confianza en modelo 1 para clase 2.")
            preprocess_gaussian1 = preprocess_gaussian(img).unsqueeze(0) #En esta línea hay un error
            print("1")
            output2 = modelMild(preprocess_gaussian1)
            print("2")
            probabilities2 = torch.nn.functional.softmax(output2, dim=1)
            confidence2, pred2 = torch.max(probabilities2, 1)
            print(output2)
            print(f"Modelo 2: Predicción {pred2.item()}, Confianza {confidence2.item():.2f}")

            # Decidir predicción final
            if confidence2.item() > 0.39:
                final_prediction = pred2.item()
            else:
                final_prediction = pred1.item()
        else:
            final_prediction = pred1.item()

        print(f"Predicción final: Clase {final_prediction}")

        descripcion = get_retinopathy_description(final_prediction)  

        return f"{descripcion}"
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    
@app.route('/verify', methods=['POST'])
def verify():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se recibió un archivo."}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No se seleccionó un archivo."}), 400

        img = Image.open(file.stream).convert('RGB')
       
        processed_img = preprocess(img).unsqueeze(0)  # Transformaciones y batch dimension

        # 4. Realizar la predicción
        with torch.no_grad():
            prediction = modelRetinoNoRetino(processed_img)

        result = prediction.numpy().tolist()
        return jsonify({"result": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400
   
  

if __name__ == '__main__':

    app.run(port=os.getenv("PORT"), debug=False)