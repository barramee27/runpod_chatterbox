# Use a stable PyTorch image that matches the model requirements
FROM runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /

# Copy requirements first to leverage Docker cache
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Install Chatterbox (allowing dependencies to resolve correctly)
RUN pip install chatterbox-tts==0.1.6

# Copy the handler script
COPY rp_handler.py /

# Pre-download and cache the model weights into the image
# This prevents a 2-minute "Cold Start" every time you use the API
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cuda')"

# Start the Serverless worker
CMD ["python3", "-u", "rp_handler.py"]
