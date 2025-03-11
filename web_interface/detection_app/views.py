# web_interface/detection_app/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import io
import yaml
import pandas as pd
import ffmpeg
import torch
from datetime import datetime
import os
import traceback
import re
import tempfile
from .ml_models import UnifiedModel, emotional_consistency
from .preprocess import preprocess_audio, preprocess_image

# Load config.yaml from project root
CONFIG_PATH = os.path.join(settings.BASE_DIR, 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained UnifiedModel from static/models/ at startup
MODEL = UnifiedModel(
    audio_input_size=config['models']['audio']['input_size'],
    image_input_size=config['models']['image']['input_size'],
    num_emotions=config['models']['emotions']['num_classes']
).to(DEVICE)

model_dir = os.path.join(settings.STATICFILES_DIRS[0], 'models')
os.makedirs(model_dir, exist_ok=True)
model_files = sorted([f for f in os.listdir(model_dir) if f.startswith('unified_epoch') and f.endswith('.pth')], reverse=True)
if model_files:
    latest_model = os.path.join(model_dir, model_files[0])
    state_dict = torch.load(latest_model, map_location=DEVICE)
    MODEL.load_state_dict(state_dict)
    print(f"Loaded UnifiedModel from {latest_model}")
else:
    raise Exception(f"No pretrained unified model found in {model_dir}")

def index(request):
    return render(request, 'index.html')

def sanitize_filename(filename):
    """Remove or replace invalid characters in filenames."""
    return re.sub(r'[^A-Za-z0-9._-]', '_', filename)

def upload_video(request):
    if request.method == 'POST':
        try:
            video_file = request.FILES['video']
            original_name = video_file.name
            video_name = sanitize_filename(original_name)  # Sanitize filename
            
            # Media paths
            video_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
            result_dir = os.path.join(settings.MEDIA_ROOT, 'results')
            log_dir = os.path.join(settings.MEDIA_ROOT, 'logs')
            os.makedirs(video_dir, exist_ok=True)
            os.makedirs(result_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            
            video_path = os.path.join(video_dir, video_name)
            result_path = os.path.join(result_dir, f"result_{video_name}.csv")
            log_path = os.path.join(log_dir, f"log_{video_name}.txt")
            
            if os.path.exists(result_path):
                return JsonResponse({'status': 'exists', 'video_name': video_name})
            
            # Save video to disk
            video_stream = io.BytesIO()
            for chunk in video_file.chunks():
                video_stream.write(chunk)
            video_stream.seek(0)
            with open(video_path, 'wb') as f:
                f.write(video_stream.getvalue())
            
            # Process video using file path
            try:
                audio_stream, frame_stream = extract_audio_and_frame(video_path)
            except Exception as ffmpeg_err:
                raise Exception(f"FFmpeg processing failed: {str(ffmpeg_err)}")
            
            try:
                audio_tensor = preprocess_audio(audio_stream, tuple(config['models']['audio']['input_size']))
                image_tensor = preprocess_image(frame_stream, tuple(config['models']['image']['input_size']))
            except Exception as preprocess_err:
                raise Exception(f"Preprocessing failed: {str(preprocess_err)}")
            
            if audio_tensor.shape[0] < 16000:
                padding = torch.zeros(16000 - audio_tensor.shape[0])
                audio_tensor = torch.cat((audio_tensor, padding), dim=0)
            audio_tensor = audio_tensor.unsqueeze(0).to(DEVICE)
            image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
            
            results = {'video_name': video_name}
            log_content = f"Processing {video_name} with UnifiedModel (Original: {original_name})\n"
            with torch.no_grad():
                MODEL.eval()
                try:
                    pred, audio_em, image_em = MODEL(audio_tensor, image_tensor)
                    consistency = emotional_consistency(audio_em, image_em).item()
                except Exception as model_err:
                    raise Exception(f"Model inference failed: {str(model_err)}")
                
                pred_value = pred.item()
                results['pred'] = pred_value
                results['consistency'] = consistency
                results['is_fake'] = 1 if pred_value > 0.3 else 0
                log_content += f"Prediction: {pred_value:.2f}\nConsistency: {consistency:.2f}\n"
                log_content += f"Result: {'Fake' if results['is_fake'] else 'Real'}\n"
            
            # Save log
            with open(log_path, 'w') as f:
                f.write(log_content)
            
            # Save results
            df = pd.DataFrame([results], columns=['video_name', 'pred', 'consistency', 'is_fake'])
            df.to_csv(result_path, index=False)
            
            return JsonResponse({'status': 'processed', 'video_name': video_name})
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"Error in upload_video: {traceback_str}")
            return JsonResponse({'status': 'error', 'error': str(e)}, status=500)
    return JsonResponse({'status': 'invalid'}, status=400)

def get_results(request, video_name):
    try:
        result_dir = os.path.join(settings.MEDIA_ROOT, 'results')
        log_dir = os.path.join(settings.MEDIA_ROOT, 'logs')
        result_path = os.path.join(result_dir, f"result_{video_name}.csv")
        log_path = os.path.join(log_dir, f"log_{video_name}.txt")
        
        df = pd.read_csv(result_path)
        result = df.to_dict(orient='records')[0]
        
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        return JsonResponse({'result': result, 'log': log_content})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def extract_audio_and_frame(video_path):
    try:
        # Probe using the file path
        probe = ffmpeg.probe(video_path, cmd='ffprobe')
        duration = float(probe['format']['duration'])
        middle_time = duration / 2
        
        # Use temporary files for FFmpeg output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
            audio_path = temp_audio_file.name
            (
                ffmpeg
                .input(video_path, format='mp4')
                .output(audio_path, format='wav', acodec='pcm_s16le', ar=16000)
                .run(quiet=True)
            )
            with open(audio_path, 'rb') as f:
                audio_stream = io.BytesIO(f.read())
            os.unlink(audio_path)  # Clean up temp file
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_frame_file:
            frame_path = temp_frame_file.name
            (
                ffmpeg
                .input(video_path, format='mp4')
                .output(frame_path, format='image2', vframes=1, ss=middle_time)
                .run(quiet=True)
            )
            with open(frame_path, 'rb') as f:
                frame_stream = io.BytesIO(f.read())
            os.unlink(frame_path)  # Clean up temp file
        
        audio_stream.seek(0)
        frame_stream.seek(0)
        return audio_stream, frame_stream
    except ffmpeg.Error as e:
        raise Exception(f"FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown FFmpeg error'}")