# Usamos una versión ligera de Python
FROM python:3.9-slim

# Instalamos librerías del sistema necesarias para que OpenCV no falle
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Carpeta de trabajo
WORKDIR /app

# Copiamos e instalamos las librerías de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos tu app.py y el modelo
COPY . .

# Exponemos el puerto que usa Hugging Face
EXPOSE 7860

# Comando para arrancar la IA
CMD ["python", "app.py"]