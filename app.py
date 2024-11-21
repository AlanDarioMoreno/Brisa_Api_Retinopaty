from flask import Flask, request, jsonify
from torchvision import transforms
import torch
from PIL import Image
from dotenv import load_dotenv
from GaussianBlur import GaussianBlur
import os
from modelInicialization import modelInicialization
from classDescription import classDescription
import warnings

#Ignora un warning de seguridad de Pytorch
warnings.filterwarnings("ignore", category=FutureWarning) 

IMAGE_SIZE = 512
RETINO_CLASS= 5
RETINO_NORETINO= 2
RETINO_FILTRO= 3
CONFIDENCE_MODEL1 = 0.98
CONFIDENCE_MODEL2 = 0.39

app = Flask(__name__)
load_dotenv()

#Se instancian los modelos
modelRetinoNoRetino= modelInicialization(os.getenv("model_name_RE_NORE"),os.getenv("local_model_path_retina_NoRetina"),RETINO_NORETINO)
modelMild= modelInicialization(os.getenv("model_name_filtroGaussian"),os.getenv("local_model_path_filtroGaussian"),RETINO_FILTRO)
modelClasification = modelInicialization(os.getenv("model_name"),os.getenv("local_model_path"),RETINO_CLASS)

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizacion de color RGB y desviacion del modelo Preentrenado RestNet50
])

preprocess_gaussian = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    GaussianBlur(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizacion de color RGB y desviacion del modelo Preentrenado RestNet50
])

@app.route('/', methods=['GET'])
def home():
    return "SERVER OK"


#Ruta de prediccion de clasificación
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se recibió un archivo."}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No se seleccionó un archivo."}), 400

        img = Image.open(file.stream).convert('RGB')
       
        processed_img = preprocess(img).unsqueeze(0) 

         # Predicción con el modelo 1
        with torch.no_grad():
            output1 = modelClasification(processed_img)
            probabilities1 = torch.nn.functional.softmax(output1, dim=1)
            confidence1, pred1 = torch.max(probabilities1, 1)

        print(f"Modelo 1: Predicción {pred1.item()}, Confianza {confidence1.item():.2f}")
        print(output1)

        # Si la predicción del modelo 1 es clase 2 con confianza menor a 98%, usar modelo 2
        if pred1.item() == 2 and confidence1.item() < CONFIDENCE_MODEL1:
            print("Evaluando con modelo 2 debido a baja confianza en modelo 1 para clase 2.")
            preprocess_gaussian1 = preprocess_gaussian(img).unsqueeze(0)
            print("1")
            output2 = modelMild(preprocess_gaussian1)
            print("2")
            probabilities2 = torch.nn.functional.softmax(output2, dim=1)
            confidence2, pred2 = torch.max(probabilities2, 1)
            print(output2)
            print(f"Modelo 2: Predicción {pred2.item()}, Confianza {confidence2.item():.2f}")

            # Decidir predicción final
            if confidence2.item() > CONFIDENCE_MODEL2:
                final_prediction = pred2.item()
            else:
                final_prediction = pred1.item()
        else:
            final_prediction = pred1.item()

        print(f"Predicción final: Clase {final_prediction}")

        descripcion = classDescription(final_prediction)  

        return f"{descripcion}"
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#Ruta de prediccion de Retina o No Retina
@app.route('/verify', methods=['POST'])
def verify():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se recibió un archivo."}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No se seleccionó un archivo."}), 400

        img = Image.open(file.stream).convert('RGB')
       
        processed_img = preprocess(img).unsqueeze(0) 

        with torch.no_grad():
            prediction = modelRetinoNoRetino(processed_img)

        result = prediction.numpy().tolist()
        return jsonify({"result": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400
   
  

if __name__ == '__main__':

    app.run(port=os.getenv("PORT"), debug=False)