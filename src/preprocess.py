# src/preprocess.py
import io
import yaml
import librosa
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import random

BASE_DIR = "/home/smd/Developments/AI-ML/deepfake_detection"

def preprocess_audio(audio_input, target_size=(1, 16000)):
    try:
        if isinstance(audio_input, (str, os.PathLike)):
            full_path = os.path.join(BASE_DIR, audio_input)
            y, sr = librosa.load(full_path, sr=16000)
        elif hasattr(audio_input, 'read'):
            y, sr = librosa.load(audio_input, sr=16000)
        else:
            raise ValueError("audio_input must be a file path or a BytesIO stream")
        y = librosa.util.fix_length(y, size=target_size[1])
        tensor = torch.tensor(y, dtype=torch.float32)  # Shape: [16000]
        return tensor
    except Exception as e:
        raise Exception(f"Failed to preprocess audio {audio_input}: {e}")

def preprocess_image(image_input, target_size=(3, 224, 224)):
    try:
        if isinstance(image_input, (str, os.PathLike)):
            full_path = os.path.join(BASE_DIR, image_input)
            image = Image.open(full_path).convert('RGB')
        elif hasattr(image_input, 'read'):
            image = Image.open(image_input).convert('RGB')
        else:
            raise ValueError("image_input must be a file path or a BytesIO stream")
        transform = transforms.Compose([
            transforms.Resize((target_size[1], target_size[2])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(image)
        return tensor
    except Exception as e:
        raise Exception(f"Failed to preprocess image {image_input}: {e}")

class DeepfakeDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        
        self.audio_base = f"{config['local']['dataset_path']}{config['local']['subdirs']['audio']}{split}/"
        self.image_base = f"{config['local']['dataset_path']}{config['local']['subdirs']['frames']}{split}/"
        
        audio_real_dir = os.path.join(BASE_DIR, f"{self.audio_base}real/")
        audio_fake_dir = os.path.join(BASE_DIR, f"{self.audio_base}fake/")
        image_real_dir = os.path.join(BASE_DIR, f"{self.image_base}real/")
        image_fake_dir = os.path.join(BASE_DIR, f"{self.image_base}fake/")
        
        # Get file lists
        self.audio_real = [f"{self.audio_base}real/{f}" for f in os.listdir(audio_real_dir) if f.endswith('.wav')]
        self.audio_fake = [f"{self.audio_base}fake/{f}" for f in os.listdir(audio_fake_dir) if f.endswith('.wav')]
        self.image_real = [f"{self.image_base}real/{f}" for f in os.listdir(image_real_dir) if f.endswith(('.jpg', '.png'))]
        self.image_fake = [f"{self.image_base}fake/{f}" for f in os.listdir(image_fake_dir) if f.endswith(('.jpg', '.png'))]
        
        # Print original counts
        print(f"{split} - Original: Real Audio = {len(self.audio_real)}, Fake Audio = {len(self.audio_fake)}, "
              f"Real Images = {len(self.image_real)}, Fake Images = {len(self.image_fake)}")
        
        # Balance the dataset
        min_real = min(len(self.audio_real), len(self.image_real))
        min_fake = min(len(self.audio_fake), len(self.image_fake))
        target_size = min(min_real, min_fake)  # Equal number of real and fake
        
        # Subsample to balance
        random.seed(42)  # For reproducibility
        self.audio_real = random.sample(self.audio_real, target_size) if len(self.audio_real) > target_size else self.audio_real
        self.audio_fake = random.sample(self.audio_fake, target_size) if len(self.audio_fake) > target_size else self.audio_fake
        self.image_real = random.sample(self.image_real, target_size) if len(self.image_real) > target_size else self.image_real
        self.image_fake = random.sample(self.image_fake, target_size) if len(self.image_fake) > target_size else self.image_fake
        
        # Combine and ensure pairing
        self.audio_files = self.audio_real + self.audio_fake
        self.image_files = self.image_real + self.image_fake
        self.labels = [0] * len(self.audio_real) + [1] * len(self.audio_fake)
        
        min_length = min(len(self.audio_files), len(self.image_files))
        self.audio_files = self.audio_files[:min_length]
        self.image_files = self.image_files[:min_length]
        self.labels = self.labels[:min_length]
        
        print(f"{split} - Balanced: Total = {len(self.labels)}, Real = {len(self.audio_real)}, Fake = {len(self.audio_fake)}")
        
        self.audio_size = (1, 16000)
        self.image_size = (3, 224, 224)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_tensor = preprocess_audio(self.audio_files[idx], self.audio_size)
        image_tensor = preprocess_image(self.image_files[idx], self.image_size)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return audio_tensor, image_tensor, label

if __name__ == "__main__":
    config_path = os.path.join(BASE_DIR, "config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset = DeepfakeDataset(config, split="train")
    print(f"Dataset size: {len(dataset)} samples")
    audio_tensor, image_tensor, label = dataset[0]
    print(f"Audio shape: {audio_tensor.shape}, Image shape: {image_tensor.shape}, Label: {label}")