# web_interface/detection_app/ml_models.py
import torch
import torch.nn as nn
import torchvision.models as models
import os
import yaml
from datetime import datetime
from .s3_utils import upload_to_s3  # Updated import

class AudioModel(nn.Module):
    def __init__(self, input_size=(1, 128, 128), feature_dim=128, num_emotions=7):
        super(AudioModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(64 * 16, 128, batch_first=True)
        self.emotion_fc = nn.Linear(128, num_emotions)
        self.feature_fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.view(batch_size, 16, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        emotion_logits = self.emotion_fc(x)
        features = self.feature_fc(x)
        return features, emotion_logits

class ImageModel(nn.Module):
    def __init__(self, feature_dim=128, num_emotions=7):
        super(ImageModel, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.emotion_fc = nn.Linear(512, num_emotions)
        self.feature_fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        emotion_logits = self.emotion_fc(x)
        features = self.feature_fc(x)
        return features, emotion_logits

class EarlyFusionModel(nn.Module):
    def __init__(self, audio_input_size=(1, 128, 128), image_input_size=(3, 224, 224), num_emotions=7):
        super(EarlyFusionModel, self).__init__()
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.image_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.joint_cnn = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 784, 256),  # Adjust based on CNN output
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, audio, image):
        audio_x = self.audio_cnn(audio)
        image_x = self.image_cnn(image)
        audio_x = torch.nn.functional.interpolate(audio_x, size=(56, 56))
        x = torch.cat((audio_x, image_x), dim=1)
        x = self.joint_cnn(x)
        x = self.fc[:-1](x)
        emotion_logits = x
        pred = self.fc[-2:](x)
        return pred, emotion_logits

class MidFusionModel(nn.Module):
    def __init__(self, feature_dim=128, num_emotions=7):
        super(MidFusionModel, self).__init__()
        self.audio_model = AudioModel(feature_dim=feature_dim, num_emotions=num_emotions)
        self.image_model = ImageModel(feature_dim=feature_dim, num_emotions=num_emotions)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, audio, image):
        audio_features, audio_emotions = self.audio_model(audio)
        image_features, image_emotions = self.image_model(image)
        x = torch.cat((audio_features, image_features), dim=1)
        x = self.fc[:-1](x)
        emotion_logits = x
        pred = self.fc[-2:](x)
        return pred, emotion_logits, audio_emotions, image_emotions

class LateFusionModel(nn.Module):
    def __init__(self, feature_dim=128, num_emotions=7):
        super(LateFusionModel, self).__init__()
        self.audio_model = AudioModel(feature_dim=feature_dim, num_emotions=num_emotions)
        self.image_model = ImageModel(feature_dim=feature_dim, num_emotions=num_emotions)
        self.audio_fc = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        self.image_fc = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, audio, image):
        audio_features, audio_emotions = self.audio_model(audio)
        image_features, image_emotions = self.image_model(image)
        audio_pred = self.audio_fc(audio_features)
        image_pred = self.image_fc(image_features)
        pred = (audio_pred + image_pred) / 2
        return pred, audio_emotions, image_emotions

def emotional_consistency(audio_emotions, image_emotions):
    return torch.nn.functional.cosine_similarity(audio_emotions, image_emotions, dim=1)

def save_to_s3_log(content, filename, config):
    s3_path = f"{config['s3']['bucket']}/{config['s3']['users']['logs']}{filename}"  # Updated to users/logs/
    upload_to_s3(content.encode('utf-8'), s3_path, is_bytes=True)
    print(f"Saved log to {s3_path}")

if __name__ == "__main__":
    # Define BASE_DIR and load config from web_interface/
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(BASE_DIR, 'config.yaml'), 'r') as f:  # Corrected path
        config = yaml.safe_load(f)

    # Generate unique timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_size = 16
    audio_input = torch.randn(batch_size, 1, 128, 128)
    image_input = torch.randn(batch_size, 3, 224, 224)
    
    # Test models (simplified for web context)
    early_model = EarlyFusionModel()
    pred, emotions = early_model(audio_input, image_input)
    print(f"Early pred: {pred.shape}")
    save_to_s3_log(f"Early pred: {pred.shape}", f"early_test_{timestamp}.txt", config)