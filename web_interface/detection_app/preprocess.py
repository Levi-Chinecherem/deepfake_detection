# web_interface/detection_app/preprocess.py
import io
import librosa
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from .s3_utils import stream_from_s3

def preprocess_audio(audio_stream, target_size=(1, 16000)):
    try:
        y, sr = librosa.load(audio_stream, sr=16000)
        y = librosa.util.fix_length(y, size=target_size[1])
        tensor = torch.tensor(y, dtype=torch.float32)  # [16000]
        return tensor
    except Exception as e:
        raise Exception(f"Failed to preprocess audio: {e}")

def preprocess_image(image_stream, target_size=(3, 224, 224)):
    try:
        image = Image.open(image_stream).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((target_size[1], target_size[2])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(image)
        return tensor
    except Exception as e:
        raise Exception(f"Failed to preprocess image: {e}")