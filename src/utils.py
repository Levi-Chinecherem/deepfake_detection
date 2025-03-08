# src/utils.py
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, auc, precision_recall_curve, confusion_matrix)
import io
import yaml
from datetime import datetime
import os

BASE_DIR = "/home/smd/Developments/AI-ML/deepfake_detection"

def monitor_memory(config, model_size_mb=100):
    mem = psutil.virtual_memory()
    total_mb = mem.total / (1024 * 1024)
    used_mb = mem.used / (1024 * 1024)
    percent = mem.percent
    
    # Log to console only, no file saving
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Total Memory: {total_mb:.2f} MB, "
          f"Used Memory: {used_mb:.2f} MB, Percent Used: {percent:.2f}%, "
          f"Model Size Estimate: {model_size_mb} MB")
    
    return percent

def adjust_batch_size(config, sample_audio_mb=10, sample_image_mb=50):
    mem = psutil.virtual_memory()
    total_mb = mem.total / (1024 * 1024)
    target_mb = total_mb * (config['memory']['target_usage_percent'] / 100)
    available_mb = target_mb - 100
    
    sample_mb = sample_audio_mb + sample_image_mb
    batch_size = max(1, int(available_mb / sample_mb))
    batch_size = min(batch_size, config['training']['batch_size'])
    
    # Log to console only, no file saving
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - Adjusted batch size: {batch_size} "
          f"(Available: {available_mb:.2f} MB, Sample: {sample_mb} MB)")
    
    return batch_size

def calculate_metrics(y_true, y_pred, y_scores, config):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_curve, precision_curve)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
    
    log = "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_to_local_log(log, f"metrics_{timestamp}.txt", config)
    return metrics, fpr, tpr, precision_curve, recall_curve

def plot_and_save(plot_type, data, config, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'line':
        for key, values in data.items():
            plt.plot(values['x'], values['y'], label=values['label'])
    elif plot_type == 'scatter':
        for key, values in data.items():
            plt.scatter(values['x'], values['y'], label=values['label'])
    elif plot_type == 'bar':
        for key, values in data.items():
            plt.bar(values['x'], values['y'], label=values['label'])
    elif plot_type == 'matrix':
        plt.matshow(data['matrix'], cmap='Blues')
        plt.colorbar()
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if plot_type != 'matrix':
        plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(BASE_DIR, f"{config['local']['outputs_path']}{config['local']['subdirs']['plots']}{filename}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, format='png', dpi=300)
    plt.close()
    print(f"Saved plot to {plot_path}")

def save_to_local_log(content, filename, config):
    log_path = os.path.join(BASE_DIR, f"{config['local']['outputs_path']}{config['local']['subdirs']['logs']}{filename}")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        f.write(content)
    print(f"Saved log to {log_path}")

if __name__ == "__main__":
    config_path = os.path.join(BASE_DIR, "config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    percent = monitor_memory(config)
    print(f"Current memory usage: {percent:.2f}%")
    
    batch_size = adjust_batch_size(config)
    print(f"Adjusted batch size: {batch_size}")
    
    y_true = [0, 1, 0, 1, 0]
    y_pred = [0, 1, 1, 1, 0]
    y_scores = [0.1, 0.9, 0.8, 0.7, 0.2]
    metrics, fpr, tpr, precision_curve, recall_curve = calculate_metrics(y_true, y_pred, y_scores, config)
    print("Metrics:", metrics)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_and_save('line', {'train': {'x': range(5), 'y': [0.5, 0.4, 0.3, 0.2, 0.1], 'label': 'Train Loss'}},
                  config, "Loss Curve", "Epoch", "Loss", f"loss_curve_{timestamp}")
    plot_and_save('scatter', {'roc': {'x': fpr, 'y': tpr, 'label': f"AUC = {metrics['roc_auc']:.2f}"}},
                  config, "ROC Curve", "False Positive Rate", "True Positive Rate", f"roc_curve_{timestamp}")
    plot_and_save('matrix', {'matrix': confusion_matrix(y_true, y_pred)},
                  config, "Confusion Matrix", "Predicted", "True", f"confusion_matrix_{timestamp}")