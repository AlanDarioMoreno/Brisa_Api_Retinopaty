from download_model import download_model
from Model import Model

def modelInicialization(modelName=None, modelPath=None, numClases=None):
    
    try:
        model = download_model(modelName,modelPath)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")

    model = Model(num_classes=numClases, model_path=modelPath)
    model.eval()  # Poner el modelo en modo evaluación

    return model

