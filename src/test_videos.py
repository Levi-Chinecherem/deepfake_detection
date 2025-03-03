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
from torch.utils.data import DataLoader
from models import EarlyFusionModel, MidFusionModel, LateFusionModel, emotional_consistency
from preprocess import preprocess_audio, preprocess_image
from utils import monitor_memory, save_to_s3_log
from s3_utils import stream_from_s3, list_s3_files, upload_to_s3

def extract_audio_and_frame(video_stream):
    """
    Extract audio (WAV) and a middle frame (JPG) from a video stream in-memory.
    
    Args:
        video_stream (io.BytesIO): Video file stream from S3.
    
    Returns:
        tuple: (audio_stream, frame_stream) as BytesIO objects.
    """
    try:
        # Probe video to get duration
        probe = ffmpeg.probe(video_stream, cmd='ffprobe')
        duration = float(probe['format']['duration'])
        middle_time = duration / 2
        
        # Extract audio
        audio_stream = io.BytesIO()
        (
            ffmpeg
            .input('pipe:', format='mp4')  # Input from stream
            .output(audio_stream, format='wav', acodec='pcm_s16le', ar=16000)
            .run(input=video_stream.getvalue(), quiet=True)
        )
        audio_stream.seek(0)
        
        # Extract middle frame
        frame_stream = io.BytesIO()
        (
            ffmpeg
            .input('pipe:', format='mp4')
            .output(frame_stream, format='image2', vframes=1, ss=middle_time)
            .run(input=video_stream.getvalue(), quiet=True)
        )
        frame_stream.seek(0)
        
        return audio_stream, frame_stream
    except ffmpeg.Error as e:
        raise Exception(f"FFmpeg error: {e.stderr.decode()}")

def process_video(s3_path, models, config, device):
    """
    Process a video file and run through models for deepfake detection.
    
    Args:
        s3_path (str): S3 path to video file.
        models (dict): Trained models {fusion_type: model}.
        config (dict): Configuration.
        device: torch.device.
    
    Returns:
        dict: Results for CSV.
    """
    start_time = time.time()
    
    # Stream video from S3
    video_stream = stream_from_s3(s3_path)
    video_name = s3_path.split('/')[-1]
    
    # Extract audio and frame
    audio_stream, frame_stream = extract_audio_and_frame(video_stream)
    
    # Preprocess
    audio_tensor = preprocess_audio(audio_stream, tuple(config['models']['audio']['input_size']))
    image_tensor = preprocess_image(frame_stream, tuple(config['models']['image']['input_size']))
    
    # Add batch dimension
    audio_tensor = audio_tensor.unsqueeze(0).to(device)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Run through models
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
    
    # Compute average prediction and binary decision
    preds = [results.get('early_pred', 0), results.get('mid_pred', 0), results.get('late_pred', 0)]
    results['avg_pred'] = sum(preds) / len([p for p in preds if p > 0])
    results['is_fake'] = 1 if results['avg_pred'] > 0.5 else 0
    results['processing_time'] = time.time() - start_time
    
    # Memory check
    mem_percent = monitor_memory(config)
    if mem_percent > config['memory']['max_usage_percent']:
        raise Exception(f"Stopping: Memory usage ({mem_percent}%) exceeds limit for {video_name}")
    
    return results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test deepfake detection on video files.")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config YAML file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models
    models_dict = {
        'early': EarlyFusionModel().to(device),
        'mid': MidFusionModel().to(device),
        'late': LateFusionModel().to(device)
    }
    
    # Load latest trained models from S3
    for fusion_type in config['training']['fusion_types']:
        model_path = sorted(list_s3_files(f"{config['s3']['bucket']}/{config['s3']['outputs_path']}"
                                         f"{config['s3']['subdirs']['models']}"), reverse=True)
        latest_model = next((p for p in model_path if fusion_type in p), None)
        if latest_model:
            print(f"Loading {fusion_type} model from {latest_model}")
            model_stream = stream_from_s3(latest_model)
            state_dict = torch.load(model_stream, map_location=device)
            models_dict[fusion_type].load_state_dict(state_dict)
        else:
            print(f"No trained {fusion_type} model found in S3")
            return
    
    # List video files in test_data/
    test_data_path = f"{config['s3']['bucket']}/test_data/"
    video_files = list_s3_files(test_data_path)
    if not video_files:
        print(f"No video files found in {test_data_path}")
        return
    
    # Process videos and collect results
    results = []
    for video_path in video_files:
        print(f"Processing {video_path}")
        result = process_video(video_path, models_dict, config, device)
        results.append(result)
    
    # Save results to CSV
    df = pd.DataFrame(results, columns=['video_name', 'early_pred', 'mid_pred', 'late_pred', 'avg_pred',
                                        'is_fake', 'mid_consistency', 'late_consistency', 'processing_time',
                                        'ground_truth', 'system_a_pred', 'system_b_pred'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_path = f"{config['s3']['bucket']}/{config['s3']['outputs_path']}results/test_results_{timestamp}.csv"
    upload_to_s3(csv_buffer.getvalue().encode('utf-8'), s3_path, is_bytes=True)
    print(f"Saved results to {s3_path}")

if __name__ == "__main__":
    main()