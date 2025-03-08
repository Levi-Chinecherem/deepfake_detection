# src/train.py
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import os
from preprocess import DeepfakeDataset
from models import AudioModel, ImageModel, EarlyFusionModel, MidFusionModel, LateFusionModel, emotional_consistency
from utils import monitor_memory, adjust_batch_size

BASE_DIR = "/home/smd/Developments/AI-ML/deepfake_detection"

def train_model(model, dataloader, optimizer, criterion, config, fusion_type, device):
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch_idx, (audio, image, label) in enumerate(dataloader):
        audio, image, label = audio.to(device), image.to(device), label.to(device).view(-1, 1)
        optimizer.zero_grad()
        
        if fusion_type == 'early':
            pred, emotion_logits = model(audio, image)
            loss = criterion(pred, label)
            log_msg = (f"Batch {batch_idx}: Loss = {loss.item():.4f}, "
                       f"Pred = {pred[0].item():.4f}, Label = {label[0].item()}")
        elif fusion_type == 'mid':
            pred, _, audio_em, image_em = model(audio, image)
            consistency = emotional_consistency(audio_em, image_em)
            loss = criterion(pred, label) + nn.MSELoss()(consistency, torch.ones_like(consistency))
            log_msg = (f"Batch {batch_idx}: Loss = {loss.item():.4f}, "
                       f"Pred = {pred[0].item():.4f}, Label = {label[0].item()}, "
                       f"Consistency = {consistency[0].item():.4f}")
        else:  # late
            pred, audio_em, image_em = model(audio, image)
            consistency = emotional_consistency(audio_em, image_em)
            loss = criterion(pred, label) + nn.MSELoss()(consistency, torch.ones_like(consistency))
            log_msg = (f"Batch {batch_idx}: Loss = {loss.item():.4f}, "
                       f"Pred = {pred[0].item():.4f}, Label = {label[0].item()}, "
                       f"Consistency = {consistency[0].item():.4f}")
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * audio.size(0)
        total_samples += audio.size(0)
        
        if batch_idx % config['logging']['log_frequency'] == 0:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - {log_msg}")
        
        mem_percent = monitor_memory(config)  # Updated to not save files
        if mem_percent > config['memory']['max_usage_percent']:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - WARNING: Memory usage ({mem_percent:.2f}%) exceeds limit, continuing training")
    
    avg_loss = total_loss / total_samples
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection models locally.")
    parser.add_argument("--config", type=str, default=os.path.join(BASE_DIR, "config/config.yaml"),
                        help="Path to config YAML file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Using device: {device}")
    
    batch_size = adjust_batch_size(config)  # Updated to not save files
    train_dataset = DeepfakeDataset(config, split="train")
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
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Starting training for {fusion_type} fusion model")
        
        model = models_dict[fusion_type]
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        
        for epoch in range(config['training']['num_epochs']):
            avg_loss = train_model(model, train_loader, optimizer, criterion, config, fusion_type, device)
            
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Epoch {epoch + 1}/{config['training']['num_epochs']}: Avg Loss = {avg_loss:.4f}")
            
            model_path = os.path.join(BASE_DIR, f"{config['local']['outputs_path']}{config['local']['subdirs']['models']}{fusion_type}_epoch_{epoch + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Saved model to {model_path}")

if __name__ == "__main__":
    main()