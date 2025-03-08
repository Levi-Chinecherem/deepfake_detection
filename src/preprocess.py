# src/preprocess.py
import io
import yaml
import librosa
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from s3_utils import stream_from_local_server, list_local_server_files

def preprocess_audio(input_data, target_size=(1, 128, 128)):
    """
    Preprocess an audio file or stream into a spectrogram tensor.
    
    Args:
        input_data (str or io.BytesIO): Local server path (str) or S3 stream (BytesIO).
        target_size (tuple): Spectrogram size (channels, height, width).
    
    Returns:
        torch.Tensor: Normalized spectrogram tensor.
    """
    try:
        if isinstance(input_data, str):
            audio_stream = stream_from_local_server(input_data)
        else:  # BytesIO from S3 (test_videos.py)
            audio_stream = input_data
        
        y, sr = librosa.load(audio_stream, sr=16000)  # Resample to 16kHz
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels_lexis = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_size[1])
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        if log_mel_spec.shape[1] != target_size[2]:
            log_mel_spec = librosa.util.fix_length(log_mel_spec, size=target_size[2], axis=1)
        
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        tensor = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        return tensor
    except Exception as e:
        raise Exception(f"Failed to preprocess audio: {e}")

def preprocess_image(input_data, target_size=(3, 224, 224)):
    """
    Preprocess an image file or stream into a normalized tensor.
    
    Args:
        input_data (str or io.BytesIO): Local server path (str) or S3 stream (BytesIO).
        target_size (tuple): Image size (channels, height, width).
    
    Returns:
        torch.Tensor: Normalized image tensor.
    """
    try:
        if isinstance(input_data, str):
            image_stream = stream_from_local_server(input_data)
        else:  # BytesIO from S3 (test_videos.py)
            image_stream = input_data
        
        image = Image.open(image_stream).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((target_size[1], target_size[2])),  # [H, W]
            transforms.ToTensor(),  # [0, 1], CHW format
        ])
        tensor = transform(image)  # [3, H, W]
        return tensor
    except Exception as e:
        raise Exception(f"Failed to preprocess image: {e}")

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for loading audio and image pairs from the local server.
    
    Args:
        config (dict): Configuration from config.yaml.
        split (str): 'train', 'test', or 'validate'.
    """
    def __init__(self, config, split):
        self.config = config
        self.split = split
        
        self.audio_base = f"{config['s3']['dataset_path']}{config['s3']['subdirs']['audio']}{split}/"
        self.image_base = f"{config['s3']['dataset_path']}{config['s3']['subdirs']['frames']}{split}/"
        
        self.audio_real = list_local_server_files(f"{self.audio_base}real/")
        self.audio_fake = list_local_server_files(f"{self.audio_base}fake/")
        self.image_real = list_local_server_files(f"{self.image_base}real/")
        self.image_fake = list_local_server_files(f"{self.image_base}fake/")
        
        self.audio_files = self.audio_real + self.audio_fake
        self.image_files = self.image_real + self.image_fake
        self.labels = [0] * len(self.audio_real) + [1] * len(self.audio_fake)
        
        min_length = min(len(self.audio_files), len(self.image_files))
        self.audio_files = self.audio_files[:min_length]
        self.image_files = self.image_files[:min_length]
        self.labels = self.labels[:min_length]
        
        self.audio_size = tuple(config['models']['audio']['input_size'])
        self.image_size = tuple(config['models']['image']['input_size'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_tensor = preprocess_audio(self.audio_files[idx], self.audio_size)
        image_tensor = preprocess_image(self.image_files[idx], self.image_size)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return audio_tensor, image_tensor, label

if __name__ == "__main__":
    with open("../config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    dataset = DeepfakeDataset(config, split="train")
    print(f"Dataset size: {len(dataset)} samples")
    
    audio_tensor, image_tensor, label = dataset[0]
    print(f"Audio shape: {audio_tensor.shape}, Image shape: {image_tensor.shape}, Label: {label}")