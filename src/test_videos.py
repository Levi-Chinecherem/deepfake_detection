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
import numpy as np
from train import UnifiedModel, emotional_consistency
from preprocess import preprocess_audio, preprocess_image
from utils import monitor_memory
import os

BASE_DIR = "/home/smd/Developments/AI-ML/deepfake_detection"

def extract_audio_and_frame(video_path):
    try:
        full_path = os.path.join(BASE_DIR, video_path)
        probe = ffmpeg.probe(full_path)
        duration = float(probe['format']['duration'])
        middle_time = duration / 2
        
        audio_stream = io.BytesIO()
        audio_out, _ = (
            ffmpeg
            .input(full_path, format='mp4')
            .output('-', format='wav', acodec='pcm_s16le', ar=16000)
            .run(quiet=True, capture_stdout=True)
        )
        audio_stream.write(audio_out)
        audio_stream.seek(0)
        
        frame_stream = io.BytesIO()
        frame_out, _ = (
            ffmpeg
            .input(full_path, format='mp4')
            .output('-', format='image2', vframes=1, ss=middle_time)
            .run(quiet=True, capture_stdout=True)
        )
        frame_stream.write(frame_out)
        frame_stream.seek(0)
        
        return audio_stream, frame_stream
    except ffmpeg.Error as e:
        raise Exception(f"FFmpeg error: {e.stderr.decode()}")

def process_video(video_path, model, config, device, threshold=0.3):
    start_time = time.time()
    
    video_name = video_path.split('/')[-1]
    audio_stream, frame_stream = extract_audio_and_frame(video_path)
    
    audio_tensor = preprocess_audio(audio_stream, tuple(config['models']['audio']['input_size']))
    image_tensor = preprocess_image(frame_stream, tuple(config['models']['image']['input_size']))
    
    if audio_tensor.shape[0] < 16000:
        padding = torch.zeros(16000 - audio_tensor.shape[0])
        audio_tensor = torch.cat((audio_tensor, padding), dim=0)
    audio_tensor = audio_tensor.unsqueeze(0).to(device)  # [1, 16000]
    image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, 224, 224]
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - {video_name} - Audio shape: {audio_tensor.shape}, Image shape: {image_tensor.shape}")
    
    results = {'video_name': video_name}
    with torch.no_grad():
        model.eval()
        pred, audio_em, image_em = model(audio_tensor, image_tensor)
        consistency = emotional_consistency(audio_em, image_em).item()
        
        pred_value = pred.item()
        results['pred'] = pred_value
        results['consistency'] = consistency
        results['is_fake'] = 1 if pred_value > threshold else 0
        results['processing_time'] = time.time() - start_time
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - {video_name} - Pred: {pred_value:.4f}, Consistency: {consistency:.4f}, Is Fake: {results['is_fake']}")
    
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
    parser = argparse.ArgumentParser(description="Test deepfake detection on local video files with UnifiedModel.")
    parser.add_argument("--config", type=str, default=os.path.join(BASE_DIR, "config/config.yaml"),
                        help="Path to config YAML file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Using device: {device}")
    
    model = UnifiedModel(
        audio_input_size=config['models']['audio']['input_size'],
        image_input_size=config['models']['image']['input_size'],
        num_emotions=config['models']['emotions']['num_classes']
    ).to(device)
    
    model_dir = f"{config['local']['outputs_path']}{config['local']['subdirs']['models']}"
    model_paths = sorted(list_local_files(model_dir), reverse=True)
    latest_model = next((p for p in model_paths if 'unified_epoch' in p), None)
    
    if latest_model:
        full_model_path = os.path.join(BASE_DIR, latest_model)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Loading unified model from {full_model_path}")
        state_dict = torch.load(full_model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - No trained unified model found locally")
        return
    
    test_data_path = config['local']['test_data_path']
    video_files = [f for f in list_local_files(test_data_path) if f.endswith('.mp4')]
    if not video_files:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - No video files found in {test_data_path}")
        return
    
    results = []
    for video_path in video_files:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Processing {video_path}")
        result = process_video(video_path, model, config, device, threshold=0.3)
        results.append(result)
    
    df = pd.DataFrame(results, columns=['video_name', 'pred', 'consistency', 'is_fake', 'processing_time'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(BASE_DIR, f"{config['local']['outputs_path']}{config['local']['subdirs']['results']}test_results_{timestamp}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Saved results to {csv_path}")

if __name__ == "__main__":
    main()