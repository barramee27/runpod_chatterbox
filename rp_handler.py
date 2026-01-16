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

# Shared variables
model = None
device = None
output_filename = "output.wav"

def optimize_torch_settings():
    """Configure PyTorch for optimal GPU performance on 24GB GPU"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Set memory fraction to leave headroom
        torch.cuda.set_per_process_memory_fraction(0.90)
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

def save_data_url_to_file(data_url, output_path="./my_audio"):
    """
    Convert a data URL (base64 encoded audio) to a temporary audio file
    
    Args:
        data_url (str): Data URL string (e.g., "data:audio/wav;base64,...")
        output_path (str): Directory to save the audio file
    
    Returns:
        str: Path to the saved audio file, or None if conversion failed
    """
    try:
        # Check if it's a data URL
        if not data_url.startswith('data:'):
            return None
        
        # Parse the data URL
        header, encoded = data_url.split(',', 1)
        
        # Extract MIME type if present
        if ';base64' in header:
            mime_type = header.split(':')[1].split(';')[0]
        else:
            mime_type = 'application/octet-stream'
        
        # Determine file extension from MIME type
        mime_to_ext = {
            'audio/wav': 'wav',
            'audio/wave': 'wav',
            'audio/x-wav': 'wav',
            'audio/mpeg': 'mp3',
            'audio/mp3': 'mp3',
            'audio/mp4': 'm4a',
            'audio/x-m4a': 'm4a',
            'audio/ogg': 'ogg',
            'audio/webm': 'webm',
            'video/mp4': 'mp4',
            'video/webm': 'webm',
            'video/x-msvideo': 'avi',
        }
        
        # Default to wav if unknown
        ext = mime_to_ext.get(mime_type, 'wav')
        
        # Create output directory if it doesn't exist
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Decode base64 data
        audio_data = base64.b64decode(encoded)
        
        # Save to temporary file
        output_file = os.path.join(output_path, f"uploaded_audio.{ext}")
        with open(output_file, 'wb') as f:
            f.write(audio_data)
        
        print(f"Saved uploaded audio file: {output_file} ({len(audio_data) / 1024:.2f} KB)")
        return output_file
        
    except Exception as e:
        print(f"Error converting data URL to file: {str(e)}")
        return None

def download_youtube_audio(url, output_path="./downloads", audio_format="mp3", duration_limit=60):
    """
    Download audio from a YouTube video
    
    Args:
        url (str): YouTube video URL
        output_path (str): Directory to save the audio file
        audio_format (str): Audio format (mp3, wav, m4a, etc.)
    
    Returns:
        tuple: (info dict, filepath) or (None, None) if failed
    """
    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/output.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
            'preferredquality': '192',
        }],
        'postprocessor_args': [
            '-ar', '44100'
        ],
        'prefer_ffmpeg': True,
    }
    if duration_limit:
        ydl_opts['postprocessors'].append({
            'key': 'FFmpegVideoConvertor',
            'preferedformat': audio_format,
        })
        ydl_opts['postprocessor_args'].extend([
            '-t', str(duration_limit)
        ])
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_duration = info.get('duration', 0)
            print(f"Title: {info.get('title', 'Unknown')}")
            print(f"Duration: {info.get('duration', 'Unknown')} seconds")
            
            if duration_limit:
                actual_duration = min(duration_limit, video_duration)
                print(f"Downloading first {actual_duration} seconds")
            
            ydl.download([url])
            print("Download completed successfully!")

            expected_filepath = os.path.join(output_path, f"output.{audio_format}")
            return info, expected_filepath
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

def audio_tensor_to_base64(audio_tensor, sample_rate):
    """Convert audio tensor to base64 encoded WAV data."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, audio_tensor, sample_rate)
            
            with open(tmp_file.name, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            os.unlink(tmp_file.name)
            return base64.b64encode(audio_data).decode('utf-8')
            
    except Exception as e:
        print(f"Error converting audio to base64: {e}")
        raise

def handler(event):
    global model, device
    input_data = event['input']
    
    # Get parameters from the request, fallback to defaults if missing
    prompt = input_data.get('prompt', '')
    yt_url = input_data.get('yt_url', '')
    exaggeration = float(input_data.get('exaggeration', 0.5))
    cfg_weight = float(input_data.get('cfg_weight', 0.5))
    temperature = float(input_data.get('temperature', 0.8))

    print(f"New request. Prompt: {prompt[:50]}...")
    
    # Log GPU memory before processing
    mem_before = get_gpu_memory_info()
    if mem_before:
        print(f"GPU Memory before: {mem_before['allocated_gb']:.2f}GB allocated, {mem_before['reserved_gb']:.2f}GB reserved ({mem_before['usage_percent']:.1f}% used)")
    
    try:
        if model is None:
            initialize_model()

        # Input handling (YouTube or Base64 Upload)
        wav_file = None
        if yt_url.startswith('data:'):
            print("Processing uploaded audio file...")
            wav_file = save_data_url_to_file(yt_url)
        else:
            print(f"Downloading from YouTube: {yt_url}")
            _, wav_file = download_youtube_audio(yt_url, output_path="./my_audio", audio_format="wav")

        if not wav_file:
            return {"error": "Failed to prepare reference audio."}

        # Clear cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Generation with Mixed Precision and full parameters
        print(f"Generating audio on device: {device}")
        start_time = time.time()
        
        with torch.cuda.amp.autocast(enabled=True):
            audio_tensor = model.generate(
                prompt,
                audio_prompt_path=wav_file,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )

        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        # Log GPU memory after generation
        mem_after = get_gpu_memory_info()
        if mem_after:
            print(f"GPU Memory after: {mem_after['allocated_gb']:.2f}GB allocated, {mem_after['reserved_gb']:.2f}GB reserved ({mem_after['usage_percent']:.1f}% used)")

        # Move to CPU for final encoding
        if isinstance(audio_tensor, torch.Tensor):
            if audio_tensor.device.type != 'cpu':
                audio_tensor = audio_tensor.cpu()
                print("Moved audio tensor to CPU for processing")
        
        # Clear GPU cache after moving to CPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        torchaudio.save(output_filename, audio_tensor, model.sr)
        
        # Cleanup
        if wav_file and os.path.exists(wav_file):
            os.remove(wav_file)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

    # Convert to base64 for response
    audio_base64 = audio_tensor_to_base64(audio_tensor, model.sr)
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Log final GPU memory
    mem_final = get_gpu_memory_info()
    if mem_final:
        print(f"GPU Memory final: {mem_final['allocated_gb']:.2f}GB allocated, {mem_final['reserved_gb']:.2f}GB reserved ({mem_final['usage_percent']:.1f}% used)")
    
    return {
        "status": "success",
        "audio_base64": audio_base64,
        "metadata": {
            "sample_rate": model.sr,
            "audio_shape": list(audio_tensor.shape) if hasattr(audio_tensor, 'shape') else None
        }
    }

def initialize_model():
    global model, device
    
    if model is not None:
        return model
    
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
        
        if gpu_memory < 20:
            print(f"WARNING: GPU has {gpu_memory:.2f}GB, recommended 24GB+ for optimal performance")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available! Falling back to CPU (this will be slow)")
    
    model_id = os.getenv("MODEL_ID", "ResembleAI/chatterbox-turbo")
    
    print(f"Initializing ChatterboxTTS model: {model_id}...")
    
    # Clear cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    mem_before_load = get_gpu_memory_info()
    if mem_before_load:
        print(f"GPU Memory before model load: {mem_before_load['free_gb']:.2f}GB free")
    
    # Initialize model - DO NOT pass device parameter (causes TypeError)
    # Load first, then move to device
    try:
        model = ChatterboxTTS.from_pretrained(model_id)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Explicitly move model to GPU device
    if torch.cuda.is_available():
        print("Moving model to GPU...")
        if hasattr(model, 'to'):
            model = model.to(device)
            print("Model moved to GPU using .to() method")
        elif hasattr(model, 'cuda'):
            model = model.cuda()
            print("Model moved to GPU using .cuda() method")
        elif hasattr(model, 'model'):
            if hasattr(model.model, 'to'):
                model.model = model.model.to(device)
                print("Model moved to GPU via model.model.to()")
            elif hasattr(model.model, 'cuda'):
                model.model = model.model.cuda()
                print("Model moved to GPU via model.model.cuda()")
        
        # Verify model is on GPU
        try:
            if hasattr(model, 'parameters'):
                first_param = next(model.parameters(), None)
                if first_param is not None:
                    param_device = first_param.device
                    print(f"Model parameter device: {param_device}")
                    if param_device.type != 'cuda':
                        print("WARNING: Model parameters are not on GPU!")
                    else:
                        print("✓ Model successfully loaded on GPU!")
            elif hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                first_param = next(model.model.parameters(), None)
                if first_param is not None:
                    param_device = first_param.device
                    print(f"Model parameter device: {param_device}")
                    if param_device.type != 'cuda':
                        print("WARNING: Model parameters are not on GPU!")
                    else:
                        print("✓ Model successfully loaded on GPU!")
        except Exception as e:
            print(f"Could not verify model device: {e}")
    
    # Enable evaluation mode for inference
    if hasattr(model, 'eval'):
        model.eval()
        print("Model set to evaluation mode")
    
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
    
    # Final cache clear after initialization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    print("=" * 60)
    print("Chatterbox TTS RunPod Handler - Optimized for 24GB GPU")
    print("=" * 60)
    initialize_model()
    print("=" * 60)
    print("Handler ready and waiting for requests...")
    print("=" * 60)
    runpod.serverless.start({'handler': handler})
