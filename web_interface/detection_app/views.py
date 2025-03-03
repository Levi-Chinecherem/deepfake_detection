# web_interface/detection_app/views.py
from django.shortcuts import render
from django.http import JsonResponse
import boto3
import io
import yaml
import pandas as pd
import ffmpeg
import torch
from datetime import datetime
import os
from dotenv import load_dotenv  # Added for .env loading
from .ml_models import EarlyFusionModel, MidFusionModel, LateFusionModel, emotional_consistency
from .preprocess import preprocess_audio, preprocess_image
from .s3_utils import stream_from_s3, list_s3_files, upload_to_s3

# Load environment variables from .env
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# S3 credentials from .env
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
BUCKET = os.getenv('S3_BUCKET')

# Load config.yaml
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Initialize S3 client with credentials
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
DEVICE = torch.device("cpu")  # Adjust for EC2; GPU optional

# Load pretrained models from users/models/ at startup
MODELS = {
    'early': EarlyFusionModel().to(DEVICE),
    'mid': MidFusionModel().to(DEVICE),
    'late': LateFusionModel().to(DEVICE)
}
for fusion_type in config['models']['fusion_types']:
    model_files = sorted(list_s3_files(f"{BUCKET}/{config['s3']['users']['models']}"), reverse=True)
    latest_model = next((p for p in model_files if fusion_type in p), None)
    if latest_model:
        model_stream = stream_from_s3(latest_model)
        state_dict = torch.load(model_stream, map_location=DEVICE)
        MODELS[fusion_type].load_state_dict(state_dict)
    else:
        raise Exception(f"No pretrained {fusion_type} model found in S3 at {config['s3']['users']['models']}")

def index(request):
    return render(request, 'index.html')

def upload_video(request):
    if request.method == 'POST':
        video_file = request.FILES['video']
        video_name = video_file.name
        
        # Check if video already processed
        data_path = f"{config['s3']['users']['data']}{video_name}"
        result_path = f"{config['s3']['users']['results']}result_{video_name}.csv"
        log_path = f"{config['s3']['users']['logs']}log_{video_name}.txt"
        
        try:
            s3_client.head_object(Bucket=BUCKET, Key=result_path)
            return JsonResponse({'status': 'exists', 'video_name': video_name})
        except:
            # Upload video
            video_stream = io.BytesIO()
            for chunk in video_file.chunks():
                video_stream.write(chunk)
            video_stream.seek(0)
            upload_to_s3(video_stream.getvalue(), f"{BUCKET}/{data_path}", is_bytes=True)
            
            # Process video with pretrained models
            try:
                audio_stream, frame_stream = extract_audio_and_frame(video_stream)
                audio_tensor = preprocess_audio(audio_stream, tuple(config['models']['audio']['input_size'])).unsqueeze(0).to(DEVICE)
                image_tensor = preprocess_image(frame_stream, tuple(config['models']['image']['input_size'])).unsqueeze(0).to(DEVICE)
                
                results = {'video_name': video_name}
                log_content = f"Processing {video_name} with pretrained models\n"
                with torch.no_grad():
                    for fusion_type, model in MODELS.items():
                        model.eval()
                        if fusion_type == 'early':
                            pred, _ = model(audio_tensor, image_tensor)
                            results['early_pred'] = pred.item()
                        elif fusion_type == 'mid':
                            pred, _, audio_em, image_em = model(audio_tensor, image_tensor)
                            consistency = emotional_consistency(audio_em, image_em).item()
                            results['mid_pred'] = pred.item()
                            results['mid_consistency'] = consistency
                        else:  # late
                            pred, audio_em, image_em = model(audio_tensor, image_tensor)
                            consistency = emotional_consistency(audio_em, image_em).item()
                            results['late_pred'] = pred.item()
                            results['late_consistency'] = consistency
                        log_content += f"{fusion_type.capitalize()} Prediction: {results[f'{fusion_type}_pred']:.2f}\n"
                
                preds = [results.get('early_pred', 0), results.get('mid_pred', 0), results.get('late_pred', 0)]
                results['avg_pred'] = sum(preds) / len([p for p in preds if p > 0])
                results['is_fake'] = 1 if results['avg_pred'] > 0.5 else 0
                log_content += f"Average Prediction: {results['avg_pred']:.2f} - {'Fake' if results['is_fake'] else 'Real'}\n"
                
                # Save log
                upload_to_s3(log_content.encode('utf-8'), f"{BUCKET}/{log_path}", is_bytes=True)
                
                # Save results
                df = pd.DataFrame([results], columns=['video_name', 'early_pred', 'mid_pred', 'late_pred', 'avg_pred', 'is_fake', 'mid_consistency', 'late_consistency'])
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                upload_to_s3(csv_buffer.getvalue().encode('utf-8'), f"{BUCKET}/{result_path}", is_bytes=True)
                
                return JsonResponse({'status': 'processed', 'video_name': video_name})
            except Exception as e:
                return JsonResponse({'status': 'error', 'error': str(e)}, status=500)
    return JsonResponse({'status': 'invalid'}, status=400)

def get_results(request, video_name):
    result_path = f"{config['s3']['users']['results']}result_{video_name}.csv"
    log_path = f"{config['s3']['users']['logs']}log_{video_name}.txt"
    try:
        result_obj = s3_client.get_object(Bucket=BUCKET, Key=result_path)
        df = pd.read_csv(io.BytesIO(result_obj['Body'].read()))
        result = df.to_dict(orient='records')[0]
        
        log_obj = s3_client.get_object(Bucket=BUCKET, Key=log_path)
        log_content = log_obj['Body'].read().decode('utf-8')
        
        return JsonResponse({'result': result, 'log': log_content})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def extract_audio_and_frame(video_stream):
    probe = ffmpeg.probe(video_stream, cmd='ffprobe')
    duration = float(probe['format']['duration'])
    middle_time = duration / 2
    audio_stream = io.BytesIO()
    (
        ffmpeg
        .input('pipe:', format='mp4')
        .output(audio_stream, format='wav', acodec='pcm_s16le', ar=16000)
        .run(input=video_stream.getvalue(), quiet=True)
    )
    audio_stream.seek(0)
    frame_stream = io.BytesIO()
    (
        ffmpeg
        .input('pipe:', format='mp4')
        .output(frame_stream, format='image2', vframes=1, ss=middle_time)
        .run(input=video_stream.getvalue(), quiet=True)
    )
    frame_stream.seek(0)
    return audio_stream, frame_stream