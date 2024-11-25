Titulo: Brisa_Api_Retinopaty

Api que contiene la implementacion de modelos de ia para deteccion de retinopatia.
Proyecto final de ORT BE, grupo 4


Descripcion:
Esta API es la encargada de descargar de la deteccion de retinopatia diabetica por medio de modelos de clasificacion de imagenes.
Estos modelos son descargados desde GCS(Google Cloud Storage) y fueron entrenados en diferentes tareas.

El primer modelo diferencia entre Retinas e Imagenes convencionales
El segundo y tercer modelo se encarga de la prediccion de retinopatia diabetica en la imagenes.


Estas tareas se llevan a cabo con la implementacion de 2 endpoints (predict y verify) 
La API se ejecuta a travez de Flask y los modelos fueron entrenados con TensorFlow.


Instrucciones paso a paso para instalar y configurar el proyecto:
Clona el repositorio:

git clone https://github.com/AlanDarioMoreno/Brisa_Api_Retinopaty.git

Tener instalado Python version: 2.17

Generar un entorno virtual para instalar las dependencias.

    python -m venv entorno

    Activar el entorno
    entorno\Scripts\activate

    Instalar las dependencias
    pip install -r requirements.txt

Es necesario tambien crear un archivo en la raiz del proyecyo con el nombre ".env"
este contendra las ubicaciones de descarga de los modelos, la ubicacion de las credenciales de GCS
y el puerto en el cual se ejecutara FLASK.

Para testear esta Api utilizamos POSTMAN

El endpoint /Verify devuelve un Array con 2 posiciones donde 
[0]>[1] es igual a una imagen que no es Retina
[0]<[1] es igual a una imagen de Retina

El endpoint /Predict devuelve alguno de los resultados declarados en classDescription.


Creditos:
    Paul Creaney
    Alan Moreno
    Gianmarco Zodda
    Franco Di Iorio
    Juan Motok



