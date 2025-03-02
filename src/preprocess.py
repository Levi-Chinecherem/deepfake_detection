# src/preprocess.py

import io
import yaml
import librosa
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from s3_utils import stream_from_s3, list_s3_files

def preprocess_audio(s3_path, target_size=(128, 128)):
    """
    Preprocess an audio file from S3 into a spectrogram tensor.
    
    Args:
        s3_path (str): S3 path to WAV file.
        target_size (tuple): Spectrogram size (height, width).
    
    Returns:
        torch.Tensor: Normalized spectrogram tensor.
    """
    try:
        # Stream audio from S3
        audio_stream = stream_from_s3(s3_path)
        
        # Load audio from stream (temp conversion to numpy array)
        y, sr = librosa.load(audio_stream, sr=16000)  # Resample to 16kHz
        
        # Generate spectrogram (log-mel)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_size[0])
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize to target size if needed
        if log_mel_spec.shape[1] != target_size[1]:
            log_mel_spec = librosa.util.fix_length(log_mel_spec, size=target_size[1], axis=1)
        
        # Normalize to [0, 1]
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        
        # Convert to tensor (add channel dimension: 1 x height x width)
        tensor = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0)
        return tensor
    except Exception as e:
        raise Exception(f"Failed to preprocess audio {s3_path}: {e}")

def preprocess_image(s3_path, target_size=(224, 224)):
    """
    Preprocess an image file from S3 into a normalized tensor.
    
    Args:
        s3_path (str): S3 path to JPG/PNG file.
        target_size (tuple): Image size (height, width).
    
    Returns:
        torch.Tensor: Normalized image tensor.
    """
    try:
        # Stream image from S3
        image_stream = stream_from_s3(s3_path)
        
        # Open image with PIL
        image = Image.open(image_stream).convert('RGB')
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),  # Converts to [0, 1] and CHW format
        ])
        
        # Apply transforms
        tensor = transform(image)
        return tensor
    except Exception as e:
        raise Exception(f"Failed to preprocess image {s3_path}: {e}")

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for loading audio and image pairs from S3.
    
    Args:
        config (dict): Configuration from config.yaml.
        split (str): 'train', 'test', or 'validate'.
    """
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.bucket = config['s3']['bucket']
        self.audio_base = f"{self.bucket}/{config['s3']['dataset_path']}{config['s3']['subdirs']['audio']}{split}/"
        self.image_base = f"{self.bucket}/{config['s3']['dataset_path']}{config['s3']['subdirs']['frames']}{split}/"
        
        # List files from S3
        self.audio_real = list_s3_files(f"{self.audio_base}real/")
        self.audio_fake = list_s3_files(f"{self.audio_base}fake/")
        self.image_real = list_s3_files(f"{self.image_base}real/")
        self.image_fake = list_s3_files(f"{self.image_base}fake/")
        
        # Combine real and fake (random pairing due to non-synchronized data)
        self.audio_files = self.audio_real + self.audio_fake
        self.image_files = self.image_real + self.image_fake
        self.labels = [0] * len(self.audio_real) + [1] * len(self.audio_fake)
        
        # Ensure equal length (trim longer modality)
        min_length = min(len(self.audio_files), len(self.image_files))
        self.audio_files = self.audio_files[:min_length]
        self.image_files = self.image_files[:min_length]
        self.labels = self.labels[:min_length]
        
        self.audio_size = tuple(config['models']['audio']['input_size'])
        self.image_size = tuple(config['models']['image']['input_size'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a preprocessed audio/image pair and label.
        
        Returns:
            tuple: (audio_tensor, image_tensor, label)
        """
        audio_tensor = preprocess_audio(self.audio_files[idx], self.audio_size)
        image_tensor = preprocess_image(self.image_files[idx], self.image_size)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return audio_tensor, image_tensor, label

if __name__ == "__main__":
    # Test the dataset
    with open("../config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    dataset = DeepfakeDataset(config, split="train")
    print(f"Dataset size: {len(dataset)} samples")
    
    # Test fetching one sample
    audio_tensor, image_tensor, label = dataset[0]
    print(f"Audio shape: {audio_tensor.shape}, Image shape: {image_tensor.shape}, Label: {label}")