Proyecto de ChatBot y Visión Computacional

Este repositorio contiene dos scripts principales:

ChatBot.py - Un sistema de chatbot que procesa archivos de audio, video, documentos y entradas de voz para responder preguntas utilizando técnicas de procesamiento de lenguaje natural (NLP) con spaCy y APIs de Google Cloud.
Vision Computacional.py - Un script para el procesamiento de imágenes que permite la selección de polígonos, segmentación de color y cálculo de áreas utilizando OpenCV y NumPy.

Características Principales
ChatBot.py

Conversión de video a audio (WAV) y procesamiento de archivos .pdf, .docx y .wav.
Tokenización de texto usando spaCy y la API de lenguaje natural de Google.
Análisis semántico para encontrar las respuestas más relevantes a preguntas de los usuarios.
Interfaz de chat para preguntas por voz o texto.

Vision Computacional.py
Selección de polígonos en imágenes mediante eventos de clic del mouse.
Segmentación de color en el espacio de color HSV.
Detección de contornos y cálculo de áreas en unidades reales (cm²).
Visualización de resultados usando matplotlib.

Requisitos Previos
Instalar las siguientes dependencias:
pip install spacy google-cloud-speech google-cloud-language PyPDF2 pyaudio keyboard docx2txt moviepy pytube pydub sklearn opencv-python numpy matplotlib
Configurar las credenciales de Google Cloud:
Crear un proyecto en Google Cloud Console.
Habilitar las APIs de Speech-to-Text y Natural Language.
Descargar el archivo JSON de credenciales y configurar la variable de entorno:

export GOOGLE_APPLICATION_CREDENTIALS="/ruta/a/credenciales.json"
Uso
ChatBot.py

Ejecutar el script para interactuar con el chatbot:
python ChatBot.py

El menú principal ofrece las siguientes opciones:

Procesar archivos de video o audio
Procesar documentos de Word o PDF
Procesar videos de YouTube
Salir del programa

Vision Computacional.py

Procesa imágenes para calcular áreas reales en función de una escala definida:
python Vision\ Computacional.py

Se solicitarán los siguientes parámetros:
Cantidad de centímetros definidos
Cantidad de píxeles equivalentes a los centímetros definidos
Altura real del objeto en cm
Contribuciones

Las contribuciones son bienvenidas. Para contribuir, por favor sigue los siguientes pasos:

Haz un fork del repositorio.
Crea una nueva rama (git checkout -b feature/nueva-funcionalidad).
Realiza los cambios y confirma los commits.
Envía un pull request.

Licencia

Este proyecto se distribuye bajo la licencia MIT. Ver el archivo LICENSE para más detalles.
