# Usamos una imagen de Python base
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Dependencias del sistema para OpenCV y YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- CAMBIO CLAVE PARA AWS GPU ---
# Instalamos la versión de Torch que soporta CUDA (GPU)
# Ya no usamos el link de /cpu
RUN pip install --no-cache-dir torch torchvision torchaudio
RUN pip install --no-cache-dir ultralytics flask flask-cors gunicorn opencv-python-headless

# Copiamos el código
COPY . .

# Exponemos el puerto
EXPOSE 8080

# Comando para producción con Gunicorn
# Nota: Quitamos el límite de memoria para que AWS use toda su potencia
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "0", "app:app"]