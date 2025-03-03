# web_interface/detection_app/preprocess.py
import io
import librosa
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from .s3_utils import stream_from_s3  # Updated import

def preprocess_audio(audio_stream, target_size=(128, 128)):
    try:
        y, sr = librosa.load(audio_stream, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_size[0])
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        if log_mel_spec.shape[1] != target_size[1]:
            log_mel_spec = librosa.util.fix_length(log_mel_spec, size=target_size[1], axis=1)
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        tensor = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0)
        return tensor
    except Exception as e:
        raise Exception(f"Failed to preprocess audio: {e}")

def preprocess_image(image_stream, target_size=(224, 224)):
    try:
        image = Image.open(image_stream).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        tensor = transform(image)
        return tensor
    except Exception as e:
        raise Exception(f"Failed to preprocess image: {e}")