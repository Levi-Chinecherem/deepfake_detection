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

BASE_DIR = "/home/smd/Developments/AI-ML/deepfake_detection"

def preprocess_audio(local_path, target_size=(1, 128, 128)):
    try:
        full_path = os.path.join(BASE_DIR, local_path)
        y, sr = librosa.load(full_path, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_size[1])
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        if log_mel_spec.shape[1] != target_size[2]:
            log_mel_spec = librosa.util.fix_length(log_mel_spec, size=target_size[2], axis=1)
        
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        tensor = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0)
        return tensor
    except Exception as e:
        raise Exception(f"Failed to preprocess audio {local_path}: {e}")

def preprocess_image(local_path, target_size=(3, 224, 224)):
    try:
        full_path = os.path.join(BASE_DIR, local_path)
        image = Image.open(full_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((target_size[1], target_size[2])),
            transforms.ToTensor(),
        ])
        tensor = transform(image)
        return tensor
    except Exception as e:
        raise Exception(f"Failed to preprocess image {local_path}: {e}")

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
        
        self.audio_real = [f"{self.audio_base}real/{f}" for f in os.listdir(audio_real_dir) if f.endswith('.wav')]
        self.audio_fake = [f"{self.audio_base}fake/{f}" for f in os.listdir(audio_fake_dir) if f.endswith('.wav')]
        self.image_real = [f"{self.image_base}real/{f}" for f in os.listdir(image_real_dir) if f.endswith(('.jpg', '.png'))]
        self.image_fake = [f"{self.image_base}fake/{f}" for f in os.listdir(image_fake_dir) if f.endswith(('.jpg', '.png'))]
        
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
    config_path = os.path.join(BASE_DIR, "config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset = DeepfakeDataset(config, split="train")
    print(f"Dataset size: {len(dataset)} samples")
    
    audio_tensor, image_tensor, label = dataset[0]
    print(f"Audio shape: {audio_tensor.shape}, Image shape: {image_tensor.shape}, Label: {label}")