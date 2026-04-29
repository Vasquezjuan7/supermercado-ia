# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies (forced CPU version to save space)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir ultralytics flask flask-cors gunicorn opencv-python-headless


# Copy the rest of the application code
COPY . .

# Expose the port (Railway uses PORT environment variable)
EXPOSE 8080

# Command to run the application using Gunicorn for production
# Workers reduced to 1 to stay within memory limits
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "0", "app:app"]