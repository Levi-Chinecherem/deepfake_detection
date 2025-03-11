# src/models.py
import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from datetime import datetime
from utils import save_to_local_log
import os

BASE_DIR = "/home/smd/Developments/AI-ML/deepfake_detection"

class AudioModel(nn.Module):
    def __init__(self, input_size=(1, 128, 128), feature_dim=128, num_emotions=7):
        super(AudioModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_size[0], 16, kernel_size=3, padding=1),
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
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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
            nn.Conv2d(audio_input_size[0], 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.image_cnn = nn.Sequential(
            nn.Conv2d(image_input_size[0], 16, kernel_size=3, padding=1),
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
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256)
        )
        self.emotion_fc = nn.Linear(256, num_emotions)
        self.pred_fc = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, audio, image):
        audio_x = self.audio_cnn(audio)
        image_x = self.image_cnn(image)
        audio_x = torch.nn.functional.interpolate(audio_x, size=(56, 56))
        x = torch.cat((audio_x, image_x), dim=1)
        x = self.joint_cnn(x)
        x = self.fc(x)
        emotion_logits = self.emotion_fc(x)
        pred = self.pred_fc(x)
        return pred, emotion_logits

class MidFusionModel(nn.Module):
    def __init__(self, audio_input_size=(1, 128, 128), image_input_size=(3, 224, 224), feature_dim=128, num_emotions=7):
        super(MidFusionModel, self).__init__()
        self.audio_model = AudioModel(input_size=audio_input_size, feature_dim=feature_dim, num_emotions=num_emotions)
        self.image_model = ImageModel(feature_dim=feature_dim, num_emotions=num_emotions)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256)
        )
        self.emotion_fc = nn.Linear(256, num_emotions)
        self.pred_fc = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, audio, image):
        audio_features, audio_emotions = self.audio_model(audio)
        image_features, image_emotions = self.image_model(image)
        x = torch.cat((audio_features, image_features), dim=1)
        x = self.fc(x)
        emotion_logits = self.emotion_fc(x)
        pred = self.pred_fc(x)
        return pred, emotion_logits, audio_emotions, image_emotions

class LateFusionModel(nn.Module):
    def __init__(self, audio_input_size=(1, 128, 128), image_input_size=(3, 224, 224), feature_dim=128, num_emotions=7):
        super(LateFusionModel, self).__init__()
        self.audio_model = AudioModel(input_size=audio_input_size, feature_dim=feature_dim, num_emotions=num_emotions)
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

def save_to_local_log(content, filename, config):
    log_path = os.path.join(BASE_DIR, f"{config['local']['outputs_path']}{config['local']['subdirs']['logs']}{filename}")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        f.write(content)
    print(f"Saved log to {log_path}")

def custom_model_summary(model, input_size, model_name, config, timestamp, device="cpu"):
    model = model.to(device)
    total_params = 0
    summary_lines = [f"Model: {model_name}", f"Input Size: {input_size}", "", "-" * 80, "Layer (type)               Output Shape         Param #", "=" * 80]
    
    def calc_params(module, memo=None):
        if memo is None:
            memo = set()
        if id(module) in memo:
            return 0
        memo.add(id(module))
        params = 0
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params += module.weight.numel()
            if module.bias is not None:
                params += module.bias.numel()
        elif isinstance(module, nn.LSTM):
            # Correct LSTM param calculation: 4 * (input_size * hidden_size + hidden_size^2 + hidden_size)
            input_size, hidden_size = module.input_size, module.hidden_size
            params += 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
        elif isinstance(module, nn.BatchNorm2d):
            params += module.weight.numel() + module.bias.numel()
        for submodule in module.children():
            params += calc_params(submodule, memo)
        return params

    def format_shape(shape):
        return f"[-1, {', '.join(map(str, shape[1:]))}]" if len(shape) > 1 else f"[-1, {shape[0]}]"

    # Collect all modules
    all_modules = [(name, module) for name, module in model.named_modules() if name != ""]

    if model_name == "AudioModel":
        x = torch.randn(1, *input_size).to(device)
        outputs = {}
        current_input = x
        cnn_output = model.cnn(current_input)
        lstm_input = cnn_output.view(1, 16, -1)
        lstm_output, _ = model.lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        outputs["cnn"] = cnn_output
        outputs["lstm"] = lstm_output
        outputs["emotion_fc"] = model.emotion_fc(lstm_output)
        outputs["feature_fc"] = model.feature_fc(lstm_output)

        cnn_outputs = [current_input]
        for layer in model.cnn:
            cnn_outputs.append(layer(cnn_outputs[-1]))

        for name, module in all_modules:
            if name == "cnn":
                continue  # Skip top-level Sequential
            elif name.startswith("cnn."):
                idx = int(name.split('.')[-1])
                output = cnn_outputs[idx + 1]
            elif name == "lstm":
                output = outputs["lstm"]
            elif name in ["emotion_fc", "feature_fc"]:
                output = outputs[name]
            else:
                continue
            params = calc_params(module)
            layer_name = f"{module.__class__.__name__}-{name.split('.')[-1]}"
            if params > 0 or isinstance(module, (nn.ReLU, nn.MaxPool2d, nn.Dropout, nn.Sigmoid, nn.Flatten)):
                summary_lines.append(f"{layer_name:<25} {format_shape(output.shape):<20} {params:,}")
                total_params += params

    elif model_name == "ImageModel":
        x = torch.randn(1, *input_size).to(device)
        outputs = {}
        backbone_output = model.backbone(x)
        flat_output = backbone_output.view(1, -1)
        outputs["backbone"] = backbone_output
        outputs["emotion_fc"] = model.emotion_fc(flat_output)
        outputs["feature_fc"] = model.feature_fc(flat_output)

        backbone_outputs = [x]
        for layer in model.backbone:
            backbone_outputs.append(layer(backbone_outputs[-1]))

        for name, module in all_modules:
            if name == "backbone":
                continue  # Skip top-level Sequential
            elif name.startswith("backbone."):
                idx = name.split('.')[-1]
                if idx.isdigit():
                    idx = int(idx)
                    output = backbone_outputs[idx + 1]
                else:
                    # Handle named layers (conv1, bn1, etc.) by finding their position
                    for i, layer in enumerate(model.backbone):
                        if layer == module:
                            output = backbone_outputs[i + 1]
                            break
            elif name in ["emotion_fc", "feature_fc"]:
                output = outputs[name]
            else:
                continue
            params = calc_params(module)
            layer_name = f"{module.__class__.__name__}-{name.split('.')[-1]}"
            if params > 0 or isinstance(module, (nn.ReLU, nn.MaxPool2d, nn.Dropout, nn.Sigmoid, nn.Flatten)):
                summary_lines.append(f"{layer_name:<25} {format_shape(output.shape):<20} {params:,}")
                total_params += params

    else:  # Fusion models
        audio_x = torch.randn(1, *input_size[0]).to(device)
        image_x = torch.randn(1, *input_size[1]).to(device)
        outputs = {}
        if model_name == "EarlyFusionModel":
            audio_cnn_out = model.audio_cnn(audio_x)
            image_cnn_out = model.image_cnn(image_x)
            audio_resized = torch.nn.functional.interpolate(audio_cnn_out, size=(56, 56))
            joint_input = torch.cat((audio_resized, image_cnn_out), dim=1)
            joint_cnn_out = model.joint_cnn(joint_input)
            fc_out = model.fc(joint_cnn_out)
            outputs["audio_cnn"] = audio_cnn_out
            outputs["image_cnn"] = image_cnn_out
            outputs["joint_cnn"] = joint_cnn_out
            outputs["fc"] = fc_out
            outputs["emotion_fc"] = model.emotion_fc(fc_out)
            outputs["pred_fc"] = model.pred_fc(fc_out)

            audio_cnn_outputs = [audio_x]
            for layer in model.audio_cnn:
                audio_cnn_outputs.append(layer(audio_cnn_outputs[-1]))
            image_cnn_outputs = [image_x]
            for layer in model.image_cnn:
                image_cnn_outputs.append(layer(image_cnn_outputs[-1]))
            joint_cnn_outputs = [joint_input]
            for layer in model.joint_cnn:
                joint_cnn_outputs.append(layer(joint_cnn_outputs[-1]))
            fc_outputs = [joint_cnn_out]
            for layer in model.fc:
                fc_outputs.append(layer(fc_outputs[-1]))
            pred_fc_outputs = [fc_out]
            for layer in model.pred_fc:
                pred_fc_outputs.append(layer(pred_fc_outputs[-1]))

        elif model_name == "MidFusionModel":
            audio_features, audio_emotions = model.audio_model(audio_x)
            image_features, image_emotions = model.image_model(image_x)
            joint_input = torch.cat((audio_features, image_features), dim=1)
            fc_out = model.fc(joint_input)
            outputs["audio_model"] = audio_features
            outputs["image_model"] = image_features
            outputs["fc"] = fc_out
            outputs["emotion_fc"] = model.emotion_fc(fc_out)
            outputs["pred_fc"] = model.pred_fc(fc_out)

            fc_outputs = [joint_input]
            for layer in model.fc:
                fc_outputs.append(layer(fc_outputs[-1]))
            pred_fc_outputs = [fc_out]
            for layer in model.pred_fc:
                pred_fc_outputs.append(layer(pred_fc_outputs[-1]))

        elif model_name == "LateFusionModel":
            audio_features, audio_emotions = model.audio_model(audio_x)
            image_features, image_emotions = model.image_model(image_x)
            outputs["audio_model"] = audio_features
            outputs["image_model"] = image_features
            outputs["audio_fc"] = model.audio_fc(audio_features)
            outputs["image_fc"] = model.image_fc(image_features)

            audio_fc_outputs = [audio_features]
            for layer in model.audio_fc:
                audio_fc_outputs.append(layer(audio_fc_outputs[-1]))
            image_fc_outputs = [image_features]
            for layer in model.image_fc:
                image_fc_outputs.append(layer(image_fc_outputs[-1]))

        for name, module in all_modules:
            if name in ["audio_cnn", "image_cnn", "joint_cnn", "fc", "pred_fc", "audio_fc", "image_fc"]:
                continue  # Skip top-level Sequential
            elif name.startswith("audio_cnn."):
                idx = int(name.split('.')[-1])
                output = audio_cnn_outputs[idx + 1]
            elif name.startswith("image_cnn."):
                idx = int(name.split('.')[-1])
                output = image_cnn_outputs[idx + 1]
            elif name.startswith("joint_cnn."):
                idx = int(name.split('.')[-1])
                output = joint_cnn_outputs[idx + 1]
            elif name.startswith("fc."):
                idx = int(name.split('.')[-1])
                output = fc_outputs[idx + 1]
            elif name.startswith("pred_fc."):
                idx = int(name.split('.')[-1])
                output = pred_fc_outputs[idx + 1]
            elif name.startswith("audio_fc."):
                idx = int(name.split('.')[-1])
                output = audio_fc_outputs[idx + 1]
            elif name.startswith("image_fc."):
                idx = int(name.split('.')[-1])
                output = image_fc_outputs[idx + 1]
            elif name in ["audio_model", "image_model", "emotion_fc"]:
                output = outputs[name]
            else:
                continue
            params = calc_params(module)
            layer_name = f"{module.__class__.__name__}-{name.split('.')[-1]}"
            if params > 0 or isinstance(module, (nn.ReLU, nn.MaxPool2d, nn.Dropout, nn.Sigmoid, nn.Flatten)):
                summary_lines.append(f"{layer_name:<25} {format_shape(output.shape):<20} {params:,}")
                total_params += params

    summary_lines.append("=" * 80)
    summary_lines.append(f"Total params: {total_params:,}")
    summary_lines.append(f"Trainable params: {total_params:,}")
    summary_lines.append("Non-trainable params: 0")
    summary_lines.append("-" * 80)
    
    log = "\n".join(summary_lines)
    print(log)
    save_to_local_log(log, f"{model_name.lower()}_summary_{timestamp}.txt", config)
    return log

if __name__ == "__main__":
    config_path = os.path.join(BASE_DIR, "config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Input sizes from config
    audio_input_size = tuple(config['models']['audio']['input_size'])  # [1, 128, 128]
    image_input_size = tuple(config['models']['image']['input_size'])  # [3, 224, 224]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # AudioModel Summary
    audio_model = AudioModel(
        input_size=audio_input_size,
        feature_dim=config['models']['audio']['output_dim'],
        num_emotions=config['models']['emotions']['num_classes']
    )
    custom_model_summary(audio_model, audio_input_size, "AudioModel", config, timestamp, device)
    
    # ImageModel Summary
    image_model = ImageModel(
        feature_dim=config['models']['image']['output_dim'],
        num_emotions=config['models']['emotions']['num_classes']
    )
    custom_model_summary(image_model, image_input_size, "ImageModel", config, timestamp, device)
    
    # EarlyFusionModel Summary
    early_model = EarlyFusionModel(
        audio_input_size=audio_input_size,
        image_input_size=image_input_size,
        num_emotions=config['models']['emotions']['num_classes']
    )
    custom_model_summary(early_model, [audio_input_size, image_input_size], "EarlyFusionModel", config, timestamp, device)
    
    # MidFusionModel Summary
    mid_model = MidFusionModel(
        audio_input_size=audio_input_size,
        image_input_size=image_input_size,
        feature_dim=config['models']['audio']['output_dim'],
        num_emotions=config['models']['emotions']['num_classes']
    )
    custom_model_summary(mid_model, [audio_input_size, image_input_size], "MidFusionModel", config, timestamp, device)
    
    # LateFusionModel Summary
    late_model = LateFusionModel(
        audio_input_size=audio_input_size,
        image_input_size=image_input_size,
        feature_dim=config['models']['audio']['output_dim'],
        num_emotions=config['models']['emotions']['num_classes']
    )
    custom_model_summary(late_model, [audio_input_size, image_input_size], "LateFusionModel", config, timestamp, device)