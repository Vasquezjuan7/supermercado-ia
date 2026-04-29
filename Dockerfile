# Usa una imagen oficial de Python ligera
FROM python:3.10-slim

# Crea y usa un directorio de trabajo
WORKDIR /app

# Instala dependencias del sistema necesarias para OpenCV
# (Este es el truco para que OpenCV no falle en la nube)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia el requirements.txt e instala las dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código
COPY . .

# Expone el puerto por defecto de HF Spaces
EXPOSE 7860

# Comando para ejecutar la aplicación con Gunicorn (más estable en la nube)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]