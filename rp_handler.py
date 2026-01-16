import runpod
import time
import torch
import torchaudio 
import yt_dlp
import os
import tempfile
import base64
import gc
from chatterbox.tts import ChatterboxTTS
from pathlib import Path

model = None
device = None
output_filename = "output.wav"

# Optimize PyTorch for 24GB GPU
def optimize_torch_settings():
    """Configure PyTorch for optimal GPU performance on 24GB GPU"""
    if torch.cuda.is_available():
        # Enable memory-efficient attention if available
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        # Set memory fraction to leave some headroom (use 90% of GPU memory)
        torch.cuda.set_per_process_memory_fraction(0.90)
        
        # Enable memory pool for faster allocations
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        print("PyTorch optimized for 24GB GPU:")
        print(f"  - TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  - Memory fraction: 90%")

def get_gpu_memory_info():
    """Get current GPU memory usage information"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - reserved,
            'usage_percent': (reserved / total) * 100
        }
    return None

def handler(event, responseFormat="base64"):
    input = event['input']    
    prompt = input.get('prompt')  
    yt_url = input.get('yt_url')  

    print(f"New request. Prompt: {prompt[:50]}...")
    
    # Log GPU memory before processing
    mem_before = get_gpu_memory_info()
    if mem_before:
        print(f"GPU Memory before: {mem_before['allocated_gb']:.2f}GB allocated, {mem_before['reserved_gb']:.2f}GB reserved ({mem_before['usage_percent']:.1f}% used)")
    
    try:
        # Ensure model is initialized
        if model is None:
            initialize_model()
        
        # Download audio from YT, cut at 60s by default
        dl_info, wav_file = download_youtube_audio(yt_url, output_path="./my_audio", audio_format="wav")
        
        if wav_file is None:
            return {"error": "Failed to download audio from YouTube URL"}

        # Clear cache before generation to maximize available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Prompt Chatterbox - ensure inputs are on correct device
        print(f"Generating audio on device: {device}")
        start_time = time.time()
        
        with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision for faster inference
            audio_tensor = model.generate(
                prompt,
                audio_prompt_path=wav_file
            )
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        # Log GPU memory after generation
        mem_after = get_gpu_memory_info()
        if mem_after:
            print(f"GPU Memory after: {mem_after['allocated_gb']:.2f}GB allocated, {mem_after['reserved_gb']:.2f}GB reserved ({mem_after['usage_percent']:.1f}% used)")
        
        # Ensure output tensor is on CPU for saving/encoding
        if isinstance(audio_tensor, torch.Tensor) and audio_tensor.device.type != 'cpu':
            audio_tensor = audio_tensor.cpu()
            print("Moved audio tensor to CPU for processing")
        
        # Clear GPU cache after moving to CPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    try:
        if os.path.exists(wav_file):
            os.remove(wav_file)
    except Exception as e:
        print(f"Warning: Could not remove temporary file {wav_file}: {e}")
    
    # Final cleanup - clear GPU cache and Python garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Log final GPU memory
    mem_final = get_gpu_memory_info()
    if mem_final:
        print(f"GPU Memory final: {mem_final['allocated_gb']:.2f}GB allocated, {mem_final['reserved_gb']:.2f}GB reserved ({mem_final['usage_percent']:.1f}% used)")

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
    
    # Optimize PyTorch settings first
    optimize_torch_settings()
    
    # Detect and set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"CUDA available! Using GPU: {gpu_name}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        print(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
        
        # Verify we have enough memory (warn if less than 20GB)
        if gpu_memory < 20:
            print(f"WARNING: GPU has {gpu_memory:.2f}GB, recommended 24GB+ for optimal performance")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available! Falling back to CPU (this will be slow)")
    
    # Get the model ID from the RunPod environment variables
    # Default to turbo if not found
    model_id = os.getenv("MODEL_ID", "ResembleAI/chatterbox-turbo")
    
    print(f"Initializing ChatterboxTTS model: {model_id}...")
    
    # Clear cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    mem_before_load = get_gpu_memory_info()
    if mem_before_load:
        print(f"GPU Memory before model load: {mem_before_load['free_gb']:.2f}GB free")
    
    # Initialize model on the detected device
    model = ChatterboxTTS.from_pretrained(model_id, device=device)
    
    # Explicitly move model to device (double-check)
    if hasattr(model, 'to'):
        model = model.to(device)
    
    # Enable evaluation mode for inference (faster, less memory)
    if hasattr(model, 'eval'):
        model.eval()
    
    # Move model to half precision if GPU has enough memory (faster inference, less memory)
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 20 * 1024**3:
        try:
            if hasattr(model, 'half'):
                model = model.half()
                print("Model converted to FP16 (half precision) for faster inference")
            elif hasattr(model, 'model') and hasattr(model.model, 'half'):
                model.model = model.model.half()
                print("Model converted to FP16 (half precision) for faster inference")
        except Exception as e:
            print(f"Warning: Could not convert to FP16: {e}. Using FP32.")
    
    mem_after_load = get_gpu_memory_info()
    if mem_after_load:
        memory_used = mem_after_load['allocated_gb'] - (mem_before_load['allocated_gb'] if mem_before_load else 0)
        print(f"GPU Memory after model load: {mem_after_load['allocated_gb']:.2f}GB allocated ({memory_used:.2f}GB used by model)")
        print(f"GPU Memory available: {mem_after_load['free_gb']:.2f}GB free")
    
    print(f"Model initialized and ready on {device}")
    
    # Verify model is on correct device
    if hasattr(model, 'device'):
        print(f"Model device: {model.device}")
    elif hasattr(model, 'model') and hasattr(model.model, 'device'):
        print(f"Model device: {model.model.device}")
    
    # Final cache clear after initialization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
    print("=" * 60)
    print("Chatterbox TTS RunPod Handler - Optimized for 24GB GPU")
    print("=" * 60)
    initialize_model()
    print("=" * 60)
    print("Handler ready and waiting for requests...")
    print("=" * 60)
    runpod.serverless.start({'handler': handler })
