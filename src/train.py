# src/train.py
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from preprocess import DeepfakeDataset
from models import AudioModel, ImageModel, EarlyFusionModel, MidFusionModel, LateFusionModel, emotional_consistency
from utils import monitor_memory, adjust_batch_size, save_to_s3_log
from s3_utils import upload_to_s3
import os

def train_model(model, dataloader, optimizer, criterion, config, fusion_type, device):
    """
    Train a single model for one epoch.
    
    Args:
        model: Model instance.
        dataloader: DataLoader with audio/image pairs.
        optimizer: Adam optimizer.
        criterion: Loss function (BCE).
        config: Configuration dict.
        fusion_type: 'early', 'mid', or 'late'.
        device: torch.device (cpu/gpu).
    
    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch_idx, (audio, image, label) in enumerate(dataloader):
        audio, image, label = audio.to(device), image.to(device), label.to(device).view(-1, 1)
        optimizer.zero_grad()
        
        if fusion_type == 'early':
            pred, emotion_logits = model(audio, image)
            loss = criterion(pred, label)
        elif fusion_type == 'mid':
            pred, _, audio_em, image_em = model(audio, image)
            consistency = emotional_consistency(audio_em, image_em)
            loss = criterion(pred, label) + nn.MSELoss()(consistency, torch.ones_like(consistency))
        else:  # late
            pred, audio_em, image_em = model(audio, image)
            consistency = emotional_consistency(audio_em, image_em)
            loss = criterion(pred, label) + nn.MSELoss()(consistency, torch.ones_like(consistency))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * audio.size(0)
        total_samples += audio.size(0)
        
        if batch_idx % config['logging']['log_frequency'] == 0:
            log = (f"Batch {batch_idx}: Loss = {loss.item():.4f}, "
                   f"Pred = {pred[0].item():.4f}, Label = {label[0].item()}")
            if fusion_type != 'early':
                log += f", Consistency = {consistency[0].item():.4f}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_to_s3_log(log, f"{fusion_type}_batch_{batch_idx}_{timestamp}.txt", config)
        
        mem_percent = monitor_memory(config)
        if mem_percent > config['memory']['max_usage_percent']:
            raise Exception(f"Stopping: Memory usage ({mem_percent}%) exceeds limit")
    
    return total_loss / total_samples

def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection models.")
    parser.add_argument("--config", type=str, default="../config/config.yaml",
                        help="Path to config YAML file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = DeepfakeDataset(config, split="train")
    batch_size = adjust_batch_size(config)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    models_dict = {
        'early': EarlyFusionModel(
            audio_input_size=config['models']['audio']['input_size'],
            image_input_size=config['models']['image']['input_size'],
            num_emotions=config['models']['emotions']['num_classes']
        ).to(device),
        'mid': MidFusionModel(
            audio_input_size=config['models']['audio']['input_size'],
            image_input_size=config['models']['image']['input_size'],
            num_emotions=config['models']['emotions']['num_classes']
        ).to(device),
        'late': LateFusionModel(
            audio_input_size=config['models']['audio']['input_size'],
            image_input_size=config['models']['image']['input_size'],
            num_emotions=config['models']['emotions']['num_classes']
        ).to(device)
    }
    
    criterion = nn.BCELoss()
    for fusion_type in config['training']['fusion_types']:
        model = models_dict[fusion_type]
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        
        print(f"Training {fusion_type} fusion model...")
        for epoch in range(config['training']['num_epochs']):
            avg_loss = train_model(model, train_loader, optimizer, criterion, config, fusion_type, device)
            
            log = f"Epoch {epoch + 1}/{config['training']['num_epochs']}: Avg Loss = {avg_loss:.4f}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_to_s3_log(log, f"{fusion_type}_epoch_{epoch + 1}_{timestamp}.txt", config)
            print(log)
            
            model_path = f"s3://{config['s3']['bucket']}/{config['s3']['outputs_path']}{config['s3']['subdirs']['models']}{fusion_type}_epoch_{epoch + 1}_{timestamp}.pth"
            torch.save(model.state_dict(), f"/tmp/temp_{fusion_type}.pth")
            upload_to_s3(f"/tmp/temp_{fusion_type}.pth", model_path)
            os.remove(f"/tmp/temp_{fusion_type}.pth")
            print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()