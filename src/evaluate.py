# src/evaluate.py
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from preprocess import DeepfakeDataset
from train import UnifiedModel, emotional_consistency
from utils import monitor_memory, adjust_batch_size, calculate_metrics, plot_and_save
from sklearn.metrics import confusion_matrix
import os

BASE_DIR = "/home/smd/Developments/AI-ML/deepfake_detection"

def evaluate_model(model, dataloader, config, device):
    model.eval()
    y_true, y_pred, y_scores, consistency_scores = [], [], [], []
    
    with torch.no_grad():
        for audio, image, label in dataloader:
            audio, image, label = audio.to(device), image.to(device), label.to(device).view(-1, 1)
            pred, audio_em, image_em = model(audio, image)
            consistency = emotional_consistency(audio_em, image_em)
            
            y_true.extend(label.cpu().numpy().flatten())
            y_pred.extend((pred > 0.5).float().detach().cpu().numpy().flatten())
            y_scores.extend(pred.detach().cpu().numpy().flatten())
            consistency_scores.extend(consistency.cpu().numpy())
            
            mem_percent = monitor_memory(config)
            if mem_percent > config['memory']['max_usage_percent']:
                raise Exception(f"Stopping: Memory usage ({mem_percent}%) exceeds limit")
    
    return y_true, y_pred, y_scores, consistency_scores

def list_local_files(directory):
    full_dir = os.path.join(BASE_DIR, directory)
    if not os.path.exists(full_dir):
        return []
    return [os.path.join(directory, f) for f in os.listdir(full_dir) if f.endswith('.pth')]

def main():
    parser = argparse.ArgumentParser(description="Evaluate unified deepfake detection model locally.")
    parser.add_argument("--config", type=str, default=os.path.join(BASE_DIR, "config/config.yaml"),
                        help="Path to config YAML file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Using device: {device}")
    
    test_dataset = DeepfakeDataset(config, split="test")
    batch_size = adjust_batch_size(config)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = UnifiedModel(
        audio_input_size=config['models']['audio']['input_size'],
        image_input_size=config['models']['image']['input_size'],
        num_emotions=config['models']['emotions']['num_classes']
    ).to(device)
    
    model_dir = f"{config['local']['outputs_path']}{config['local']['subdirs']['models']}"
    model_paths = sorted(list_local_files(model_dir), reverse=True)
    latest_model = next((p for p in model_paths if 'unified_epoch' in p), None)
    
    if latest_model:
        full_model_path = os.path.join(BASE_DIR, latest_model)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Loading unified model from {full_model_path}")
        state_dict = torch.load(full_model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - No trained unified model found locally")
        return
    
    y_true, y_pred, y_scores, consistency_scores = evaluate_model(model, test_loader, config, device)
    
    metrics, fpr, tpr, precision_curve, recall_curve = calculate_metrics(y_true, y_pred, y_scores, config)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Unified Model Metrics: {metrics}")
    
    # Print confusion matrix raw values for debugging
    cm = confusion_matrix(y_true, y_pred)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Confusion Matrix:\n{cm}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "evaluate_unified"  # Updated prefix
    
    # Plots with evaluate_ prefix
    plot_and_save('scatter', {'roc': {'x': fpr, 'y': tpr, 'label': f"AUC = {metrics['roc_auc']:.2f}"}},
                  config, "ROC Curve", "False Positive Rate", "True Positive Rate", f"{prefix}_roc_curve_{timestamp}")
    
    plot_and_save('scatter', {'pr': {'x': recall_curve, 'y': precision_curve, 'label': f"AUC = {metrics['pr_auc']:.2f}"}},
                  config, "Precision-Recall Curve", "Recall", "Precision", f"{prefix}_pr_curve_{timestamp}")
    
    plot_and_save('matrix', {'matrix': cm},
                  config, "Confusion Matrix", "Predicted", "True", f"{prefix}_confusion_matrix_{timestamp}")
    
    plot_and_save('bar', {'dist': {'x': range(len(y_scores)), 'y': y_scores, 'label': 'Prediction Scores'}},
                  config, "Prediction Distribution", "Sample", "Score", f"{prefix}_pred_dist_{timestamp}")
    
    if consistency_scores:
        plot_and_save('bar', {'consistency': {'x': range(len(consistency_scores)), 'y': consistency_scores, 'label': 'Consistency'}},
                      config, "Emotional Consistency Histogram", "Sample", "Cosine Similarity", f"{prefix}_consistency_hist_{timestamp}")

if __name__ == "__main__":
    main()