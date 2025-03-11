# web_interface/detection_app/ml_models.py
import torch
import torch.nn as nn
import torchaudio
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class UnifiedModel(nn.Module):
    def __init__(self, audio_input_size=(1, 16000), image_input_size=(3, 224, 224), num_emotions=7):
        super(UnifiedModel, self).__init__()
        # Audio backbone (Wav2Vec2) - Match train.py
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.audio_backbone = bundle.get_model()
        self.audio_fc = nn.Linear(768, 256)  # Match train.py
        
        # Image backbone (ResNet18) - Match train.py
        self.image_backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.image_backbone.fc = nn.Linear(self.image_backbone.fc.in_features, 256)  # Replace final FC
        
        # Fusion layers - Match train.py
        self.early_fusion = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.mid_fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Emotion classification - Match train.py
        self.audio_emotion = nn.Linear(256, num_emotions)
        self.image_emotion = nn.Linear(256, num_emotions)
        
        # Late fusion - Match train.py
        self.late_fusion = nn.Sequential(
            nn.Linear(64 + num_emotions * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, audio, image):
        # Audio processing
        audio_features = self.audio_backbone.extract_features(audio, num_layers=12)[0][-1].mean(dim=1)  # [batch, 768]
        audio_embed = self.audio_fc(audio_features)  # [batch, 256]
        audio_em = self.audio_emotion(audio_embed)   # [batch, num_emotions]
        
        # Image processing
        image_embed = self.image_backbone(image)     # [batch, 256]
        image_em = self.image_emotion(image_embed)   # [batch, num_emotions]
        
        # Fusion
        early = torch.cat((audio_embed, image_embed), dim=1)  # [batch, 512]
        early_out = self.early_fusion(early)                  # [batch, 128]
        mid_out = self.mid_fusion(early_out)                  # [batch, 64]
        late_in = torch.cat((mid_out, audio_em, image_em), dim=1)  # [batch, 64 + 2*num_emotions]
        pred = self.late_fusion(late_in)                      # [batch, 1]
        
        return pred, audio_em, image_em

def emotional_consistency(audio_emotions, image_emotions):
    return torch.nn.functional.cosine_similarity(audio_emotions, image_emotions, dim=1)