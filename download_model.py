from google.cloud import storage
import torch
import os
import io
from dotenv import load_dotenv

def download_model(modelName=None, localPath= None):
    load_dotenv()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")   

    storage_client = storage.Client()

    bucket = storage_client.bucket(os.getenv("bucket_name"))
    blob = bucket.blob(modelName)
    blob.download_to_filename(localPath)

    # Usar un buffer en memoria para descargar
    model_buffer = io.BytesIO()
    blob.download_to_file(model_buffer)
    model_buffer.seek(0)  # Volver al inicio del buffer

    model = torch.load(os.getenv(localPath),map_location=torch.device('cpu'))
    model.eval() 

    print("Modelo cargado correctamente.")

    return model