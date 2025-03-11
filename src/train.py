# src/train.py
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import os
import numpy as np
import torchaudio
from torchvision import models
from torchvision.models import ResNet18_Weights
from preprocess import DeepfakeDataset
from utils import monitor_memory, adjust_batch_size, calculate_metrics, plot_and_save
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import glob

BASE_DIR = "/home/smd/Developments/AI-ML/deepfake_detection"

class UnifiedModel(nn.Module):
    def __init__(self, audio_input_size, image_input_size, num_emotions):
        super(UnifiedModel, self).__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.audio_backbone = bundle.get_model()
        self.audio_fc = nn.Linear(768, 256)
        self.image_backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.image_backbone.fc = nn.Linear(self.image_backbone.fc.in_features, 256)
        self.early_fusion = nn.Sequential(nn.Linear(256 + 256, 128), nn.ReLU(), nn.Dropout(0.3))
        self.mid_fusion = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))
        self.audio_emotion = nn.Linear(256, num_emotions)
        self.image_emotion = nn.Linear(256, num_emotions)
        self.late_fusion = nn.Sequential(nn.Linear(64 + num_emotions * 2, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1), nn.Sigmoid())
    
    def forward(self, audio, image):
        audio_features = self.audio_backbone.extract_features(audio, num_layers=12)[0][-1].mean(dim=1)
        audio_embed = self.audio_fc(audio_features)
        audio_em = self.audio_emotion(audio_embed)
        image_embed = self.image_backbone(image)
        image_em = self.image_emotion(image_embed)
        early = torch.cat((audio_embed, image_embed), dim=1)
        early_out = self.early_fusion(early)
        mid_out = self.mid_fusion(early_out)
        late_in = torch.cat((mid_out, audio_em, image_em), dim=1)
        pred = self.late_fusion(late_in)
        return pred, audio_em, image_em

def emotional_consistency(audio_em, image_em):
    return torch.cosine_similarity(audio_em, image_em)

def train_model(model, dataloader, optimizer, criterion, config, device):
    model.train()
    total_loss = 0
    total_samples = 0
    all_preds, all_scores, all_labels = [], [], []
    
    for batch_idx, (audio, image, label) in enumerate(dataloader):
        audio, image, label = audio.to(device), image.to(device), label.to(device).view(-1, 1)
        optimizer.zero_grad()
        pred, audio_em, image_em = model(audio, image)
        consistency = emotional_consistency(audio_em, image_em)
        loss = criterion(pred, label) + 2.0 * nn.MSELoss()(consistency, torch.ones_like(consistency))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * audio.size(0)
        total_samples += audio.size(0)
        
        # Detach tensors before converting to NumPy
        pred_binary = (pred > 0.5).float().detach().cpu().numpy()
        pred_scores = pred.detach().cpu().numpy()
        label_np = label.detach().cpu().numpy()
        all_preds.extend(pred_binary.flatten())
        all_scores.extend(pred_scores.flatten())
        all_labels.extend(label_np.flatten())
        
        if batch_idx % config['logging']['log_frequency'] == 0:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / total_samples
    return avg_loss, all_preds, all_scores, all_labels

def validate_model(model, dataloader, criterion, config, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds, all_scores, all_labels = [], [], []
    all_audio_embeds, all_image_embeds = [], []
    
    with torch.no_grad():
        for audio, image, label in dataloader:
            audio, image, label = audio.to(device), image.to(device), label.to(device).view(-1, 1)
            pred, audio_em, image_em = model(audio, image)
            consistency = emotional_consistency(audio_em, image_em)
            loss = criterion(pred, label) + 2.0 * nn.MSELoss()(consistency, torch.ones_like(consistency))
            total_loss += loss.item() * audio.size(0)
            total_samples += audio.size(0)
            
            # Detach for consistency, though no_grad() might suffice
            all_preds.extend((pred > 0.5).float().detach().cpu().numpy().flatten())
            all_scores.extend(pred.detach().cpu().numpy().flatten())
            all_labels.extend(label.detach().cpu().numpy().flatten())
            all_audio_embeds.extend(audio_em.detach().cpu().numpy())
            all_image_embeds.extend(image_em.detach().cpu().numpy())
    
    avg_loss = total_loss / total_samples
    return avg_loss, all_preds, all_scores, all_labels, np.array(all_audio_embeds), np.array(all_image_embeds)

def main():
    parser = argparse.ArgumentParser(description="Train unified deepfake detection model.")
    parser.add_argument("--config", type=str, default=os.path.join(BASE_DIR, "config/config.yaml"))
    parser.add_argument("--force-retrain", action="store_true", help="Force retrain from scratch")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Using device: {device}")
    
    batch_size = adjust_batch_size(config)
    train_dataset = DeepfakeDataset(config, split="train")
    val_dataset = DeepfakeDataset(config, split="validate")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = UnifiedModel(
        audio_input_size=config['models']['audio']['input_size'],
        image_input_size=config['models']['image']['input_size'],
        num_emotions=config['models']['emotions']['num_classes']
    ).to(device)
    
    start_epoch = 0
    if not args.force_retrain:
        model_dir = os.path.join(BASE_DIR, f"{config['local']['outputs_path']}{config['local']['subdirs']['models']}")
        checkpoints = glob.glob(f"{model_dir}unified_epoch_*_*.pth")
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_epoch_')[1].split('_')[0]))
            model.load_state_dict(torch.load(latest_checkpoint))
            start_epoch = int(latest_checkpoint.split('_epoch_')[1].split('_')[0])
            print(f"Loaded {latest_checkpoint} at epoch {start_epoch}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.BCELoss()
    num_epochs = 20
    
    train_losses, val_losses, val_accuracies = [], [], []
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_preds, train_scores, train_labels = train_model(model, train_loader, optimizer, criterion, config, device)
        val_loss, val_preds, val_scores, val_labels, audio_embeds, image_embeds = validate_model(model, val_loader, criterion, config, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_acc = sum(1 for p, l in zip(val_preds, val_labels) if p == l) / len(val_labels)
        val_accuracies.append(val_acc)
        
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Epoch {epoch + 1}/{num_epochs}: "
              f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
        
        model_path = os.path.join(BASE_DIR, f"{config['local']['outputs_path']}{config['local']['subdirs']['models']}unified_epoch_{epoch + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    epochs_range = range(1, num_epochs + 1)
    plot_and_save('line', 
                  {'train': {'x': epochs_range, 'y': train_losses, 'label': 'Train Loss'},
                   'val': {'x': epochs_range, 'y': val_losses, 'label': 'Val Loss'}},
                  config, "Unified Loss Curve", "Epoch", "Loss", f"unified_loss_curve_{timestamp}")
    plot_and_save('line', 
                  {'val_acc': {'x': epochs_range, 'y': val_accuracies, 'label': 'Validation Accuracy'}},
                  config, "Unified Accuracy Curve", "Epoch", "Accuracy", f"unified_accuracy_curve_{timestamp}")
    metrics, fpr, tpr, precision, recall = calculate_metrics(val_labels, val_preds, val_scores, config)
    print(f"Metrics: {metrics}")
    plot_and_save('scatter', 
                  {'roc': {'x': fpr, 'y': tpr, 'label': f"AUC = {metrics['roc_auc']:.2f}"}},
                  config, "Unified ROC Curve", "FPR", "TPR", f"unified_roc_curve_{timestamp}")
    plot_and_save('line', 
                  {'pr': {'x': recall, 'y': precision, 'label': 'Precision-Recall'}},
                  config, "Unified PR Curve", "Recall", "Precision", f"unified_pr_curve_{timestamp}")
    cm = confusion_matrix(val_labels, val_preds)
    plot_and_save('matrix', 
                  {'matrix': cm},
                  config, "Unified Confusion Matrix", "Predicted", "True", f"unified_confusion_matrix_{timestamp}")
    embeds = np.concatenate([audio_embeds, image_embeds], axis=1)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeds = tsne.fit_transform(embeds)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_embeds[:, 0], tsne_embeds[:, 1], c=val_labels, cmap='coolwarm', alpha=0.5)
    plt.colorbar(scatter, label='Label (0=Real, 1=Fake)')
    plt.title("t-SNE of Unified Model Features")
    plt.savefig(os.path.join(BASE_DIR, f"{config['local']['outputs_path']}{config['local']['subdirs']['plots']}unified_tsne_{timestamp}.png"))
    plt.close()

if __name__ == "__main__":
    main()