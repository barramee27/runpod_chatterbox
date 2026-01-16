# Use the verified base image that successfully loaded in your previous build
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /

# Copy requirements first (to fix the omegaconf error)
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Force install the specific Chatterbox version and its requirements
RUN pip install chatterbox-tts==0.1.6

# Copy the handler script
COPY rp_handler.py /

# Pre-download the 1.92GB model weights into the image to avoid "Cold Start" delays
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cuda')"

# Start the Serverless worker
CMD ["python3", "-u", "rp_handler.py"]
