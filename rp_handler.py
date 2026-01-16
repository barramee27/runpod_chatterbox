import runpod
import time  
import torch
import torchaudio 
import yt_dlp
import os
import tempfile
import base64
import time
from chatterbox.tts import ChatterboxTTS
from pathlib import Path

model = None
device = None
output_filename = "output.wav"

def handler(event, responseFormat="base64"):
    input = event['input']    
    prompt = input.get('prompt')  
    yt_url = input.get('yt_url')  

    print(f"New request. Prompt: {prompt}")
    
    try:
        # Ensure model is initialized
        if model is None:
            initialize_model()
        
        # Download audio from YT, cut at 60s by default
        dl_info, wav_file = download_youtube_audio(yt_url, output_path="./my_audio", audio_format="wav")

        # Prompt Chatterbox - ensure inputs are on correct device
        print(f"Generating audio on device: {device}")
        audio_tensor = model.generate(
            prompt,
            audio_prompt_path=wav_file
        )
        
        # Ensure output tensor is on CPU for saving/encoding
        if isinstance(audio_tensor, torch.Tensor) and audio_tensor.device.type != 'cpu':
            audio_tensor = audio_tensor.cpu()
            print("Moved audio tensor to CPU for processing")

        # Save as WAV
        torchaudio.save(output_filename, audio_tensor, model.sr)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"{e}" 

    # Convert to base64 string
    audio_base64 = audio_tensor_to_base64(audio_tensor, model.sr)

    if responseFormat == "base64":
        # Return base64
        response = {
            "status": "success",
            "audio_base64": audio_base64,
            "metadata": {
                "sample_rate": model.sr,
                "audio_shape": list(audio_tensor.shape)
            }
        }
    elif responseFormat == "binary":
        with open(output_filename, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Clean up the file
        os.remove(output_filename)
        
        response = audio_data  # Just return the base64 string

    # Clean up temporary files
    os.remove(wav_file)

    return response 

def audio_tensor_to_base64(audio_tensor, sample_rate):
    """Convert audio tensor to base64 encoded WAV data."""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, audio_tensor, sample_rate)
            
            # Read back as binary data
            with open(tmp_file.name, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            # Encode as base64
            return base64.b64encode(audio_data).decode('utf-8')
            
    except Exception as e:
        print(f"Error converting audio to base64: {e}")
        raise


def initialize_model():
    global model, device
    
    if model is not None:
        return model
    
    # Detect and set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available! Falling back to CPU (this will be slow)")
    
    # Get the model ID from the RunPod environment variables
    # Default to turbo if not found
    model_id = os.getenv("MODEL_ID", "ResembleAI/chatterbox-turbo")
    
    print(f"Initializing ChatterboxTTS model: {model_id}...")
    # Initialize model on the detected device
    model = ChatterboxTTS.from_pretrained(model_id, device=device)
    
    # Explicitly move model to device (double-check)
    if hasattr(model, 'to'):
        model = model.to(device)
    
    print(f"Model initialized and ready on {device}")
    
    # Verify model is on correct device
    if hasattr(model, 'device'):
        print(f"Model device: {model.device}")
    elif hasattr(model, 'model') and hasattr(model.model, 'device'):
        print(f"Model device: {model.model.device}")
    
def download_youtube_audio(url, output_path="./downloads", audio_format="mp3", duration_limit=60):
    """
    Download audio from a YouTube video
    
    Args:
        url (str): YouTube video URL
        output_path (str): Directory to save the audio file
        audio_format (str): Audio format (mp3, wav, m4a, etc.)
    
    Returns:
        str: Path to the downloaded audio file, or None if download failed
    """
    
    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',  # Download best quality audio
        'outtmpl': f'{output_path}/output.%(ext)s',  # Output filename template
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
            'preferredquality': '192',  # Audio quality in kbps
        }],
        'postprocessor_args': [
            '-ar', '44100'  # Set sample rate
        ],
        'prefer_ffmpeg': True,
    }
    if duration_limit:
        ydl_opts['postprocessors'].append({
            'key': 'FFmpegVideoConvertor',
            'preferedformat': audio_format,
        })
        # Add FFmpeg arguments for trimming
        ydl_opts['postprocessor_args'].extend([
            '-t', str(duration_limit)  # Trim to specified duration
        ])
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            video_duration = info.get('duration', 0)
            print(f"Title: {info.get('title', 'Unknown')}")
            print(f"Duration: {info.get('duration', 'Unknown')} seconds")
            print(f"Uploader: {info.get('uploader', 'Unknown')}")
        
            if duration_limit:
                actual_duration = min(duration_limit, video_duration)
                print(f"Downloading first {actual_duration} seconds")
            
            # Download the audio
            print("Downloading audio...")
            ydl.download([url])
            print("Download completed successfully!")

            expected_filepath = os.path.join(output_path, f"output.{audio_format}")
            
            return info, expected_filepath
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == '__main__':
    initialize_model()
    runpod.serverless.start({'handler': handler })
