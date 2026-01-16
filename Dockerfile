FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y git wget curl ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /

# 1. Force remove the pre-installed conflicting versions
RUN pip uninstall -y torch torchvision torchaudio

# 2. Install the synchronized stable stack
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# 3. Install remaining requirements and the model
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
RUN pip install chatterbox-tts==0.1.6

COPY rp_handler.py /

# Change this line (likely line 26) in your Dockerfile:
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')"

CMD ["python3", "-u", "rp_handler.py"]
