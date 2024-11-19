from flask import Flask, request, jsonify
from torchvision import transforms
import torch
import cv2
from PIL import Image
from dotenv import load_dotenv
import os
from Model import Model
from modelInicialization import modelInicialization
import warnings

#Ignora un warning de seguridad de Pytorch
warnings.filterwarnings("ignore", category=FutureWarning) 

IMAGE_SIZE = 512
RETINO_CLASS= 5
RETINO_NORETINO= 2


app = Flask(__name__)
load_dotenv()

#Se instancian los modelos de clasificacion y retina/No Retina
modelClasification = modelInicialization(os.getenv("model_name"),os.getenv("local_model_path"),RETINO_CLASS)
modelRetinoNoRetino= modelInicialization(os.getenv("model_name_RE_NORE"),os.getenv("local_model_path_retina_NoRetina"),RETINO_NORETINO)

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizacion de color RGB y desviacion del modelo RestNet50
])

# Filtro Gaussian Blur
def add_gaussian_blur(image, sigmaX=10):   
    return cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

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

        #PARA APLICAR FILTRO
        #img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        #blurred_image = add_gaussian_blur(img_array)
        #blurred_image = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))

        # 3. Aplicar preprocesamiento
        processed_img = preprocess(img).unsqueeze(0)  # Transformaciones y batch dimension

        # 4. Realizar la predicción
        with torch.no_grad():
            prediction = modelClasification(processed_img)

        result = prediction.numpy().tolist()
        return jsonify({"result": result})
    
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

    app.run(port=os.getenv("PORT"), debug=True)