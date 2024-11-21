import numpy as np
import cv2
from PIL import Image

class GaussianBlur:
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