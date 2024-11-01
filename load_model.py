from google.cloud import storage
import tensorflow as tf
import os
from dotenv import load_dotenv
import os

def load_model():
    load_dotenv()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")   

    storage_client = storage.Client()

        # Accede al bucket y descarga el archivo
    bucket = storage_client.bucket(os.getenv("bucket_name"))
    blob = bucket.blob(os.getenv("model_name"))
    blob.download_to_filename(os.getenv("local_model_path"))

        # Carga el modelo descargado en TensorFlow
    model = tf.keras.models.load_model(os.getenv("local_model_path"))
    print("Modelo cargado correctamente.")

    return model