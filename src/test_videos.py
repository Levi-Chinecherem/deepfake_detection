# src/test_videos.py
import argparse
import yaml
import torch
from datetime import datetime
import time
import ffmpeg
import io
import pandas as pd
from PIL import Image
import librosa
from models import EarlyFusionModel, MidFusionModel, LateFusionModel, emotional_consistency
from preprocess import preprocess_audio, preprocess_image
from utils import monitor_memory, save_to_local_log
import os

BASE_DIR = "/home/smd/Developments/AI-ML/deepfake_detection"

def extract_audio_and_frame(video_path):
    try:
        full_path = os.path.join(BASE_DIR, video_path)
        probe = ffmpeg.probe(full_path, cmd='ffprobe')
        duration = float(probe['format']['duration'])
        middle_time = duration / 2
        
        audio_stream = io.BytesIO()
        (
            ffmpeg
            .input(full_path, format='mp4')
            .output(audio_stream, format='wav', acodec='pcm_s16le', ar=16000)
            .run(quiet=True)
        )
        audio_stream.seek(0)
        
        frame_stream = io.BytesIO()
        (
            ffmpeg
            .input(full_path, format='mp4')
            .output(frame_stream, format='image2', vframes=1, ss=middle_time)
            .run(quiet=True)
        )
        frame_stream.seek(0)
        
        return audio_stream, frame_stream
    except ffmpeg.Error as e:
        raise Exception(f"FFmpeg error: {e.stderr.decode()}")

def process_video(video_path, models, config, device):
    start_time = time.time()
    
    video_name = video_path.split('/')[-1]
    audio_stream, frame_stream = extract_audio_and_frame(video_path)
    
    audio_tensor = preprocess_audio(audio_stream, tuple(config['models']['audio']['input_size']))
    image_tensor = preprocess_image(frame_stream, tuple(config['models']['image']['input_size']))
    
    audio_tensor = audio_tensor.unsqueeze(0).to(device)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    results = {'video_name': video_name}
    with torch.no_grad():
        for fusion_type, model in models.items():
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
    
    preds = [results.get('early_pred', 0), results.get('mid_pred', 0), results.get('late_pred', 0)]
    results['avg_pred'] = sum(preds) / len([p for p in preds if p > 0])
    results['is_fake'] = 1 if results['avg_pred'] > 0.5 else 0
    results['processing_time'] = time.time() - start_time
    
    mem_percent = monitor_memory(config)
    if mem_percent > config['memory']['max_usage_percent']:
        raise Exception(f"Stopping: Memory usage ({mem_percent}%) exceeds limit for {video_name}")
    
    return results

def list_local_files(directory):
    full_dir = os.path.join(BASE_DIR, directory)
    if not os.path.exists(full_dir):
        return []
    return [os.path.join(directory, f) for f in os.listdir(full_dir) if f.endswith(('.pth', '.mp4'))]

def main():
    parser = argparse.ArgumentParser(description="Test deepfake detection on local video files.")
    parser.add_argument("--config", type=str, default=os.path.join(BASE_DIR, "config/config.yaml"),
                        help="Path to config YAML file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    models_dict = {
        'early': EarlyFusionModel(
            audio_input_size=config['models']['audio']['input_size'],
            image_input_size=config['models']['image']['input_size'],
            num_emotions=config['models']['emotions']['num_classes']
        ).to(device),
        'mid': MidFusionModel(
            audio_input_size=config['models']['audio']['input_size'],
            image_input_size=config['models']['image']['input_size'],
            num_emotions=config['models']['emotions']['num_classes']
        ).to(device),
        'late': LateFusionModel(
            audio_input_size=config['models']['audio']['input_size'],
            image_input_size=config['models']['image']['input_size'],
            num_emotions=config['models']['emotions']['num_classes']
        ).to(device)
    }
    
    for fusion_type in config['training']['fusion_types']:
        model_dir = f"{config['local']['outputs_path']}{config['local']['subdirs']['models']}"
        model_path = sorted(list_local_files(model_dir), reverse=True)
        latest_model = next((p for p in model_path if fusion_type in p), None)
        if latest_model:
            full_model_path = os.path.join(BASE_DIR, latest_model)
            print(f"Loading {fusion_type} model from {full_model_path}")
            state_dict = torch.load(full_model_path, map_location=device)
            models_dict[fusion_type].load_state_dict(state_dict)
        else:
            print(f"No trained {fusion_type} model found locally")
            return
    
    test_data_path = config['local']['test_data_path']
    video_files = [f for f in list_local_files(test_data_path) if f.endswith('.mp4')]
    if not video_files:
        print(f"No video files found in {test_data_path}")
        return
    
    results = []
    for video_path in video_files:
        print(f"Processing {video_path}")
        result = process_video(video_path, models_dict, config, device)
        results.append(result)
    
    df = pd.DataFrame(results, columns=['video_name', 'early_pred', 'mid_pred', 'late_pred', 'avg_pred',
                                        'is_fake', 'mid_consistency', 'late_consistency', 'processing_time'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(BASE_DIR, f"{config['local']['outputs_path']}{config['local']['subdirs']['results']}test_results_{timestamp}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

if __name__ == "__main__":
    main()