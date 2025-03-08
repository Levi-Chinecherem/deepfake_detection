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
from s3_utils import upload_to_s3

def monitor_memory(config, model_size_mb=100):
    mem = psutil.virtual_memory()
    total_mb = mem.total / (1024 * 1024)
    used_mb = mem.used / (1024 * 1024)
    percent = mem.percent
    
    log = (f"Total Memory: {total_mb:.2f} MB\n"
           f"Used Memory: {used_mb:.2f} MB\n"
           f"Percent Used: {percent:.2f}%\n"
           f"Model Size Estimate: {model_size_mb} MB")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_to_s3_log(log, f"memory_usage_{timestamp}.txt", config)
    
    if percent > config['memory']['max_usage_percent']:
        raise Exception(f"Memory usage ({percent}%) exceeds max limit ({config['memory']['max_usage_percent']}%)")
    return percent

def adjust_batch_size(config, sample_audio_mb=10, sample_image_mb=50):
    mem = psutil.virtual_memory()
    total_mb = mem.total / (1024 * 1024)
    target_mb = total_mb * (config['memory']['target_usage_percent'] / 100)
    available_mb = target_mb - 100
    
    sample_mb = sample_audio_mb + sample_image_mb
    batch_size = max(1, int(available_mb / sample_mb))
    batch_size = min(batch_size, config['training']['batch_size'])
    
    log = f"Adjusted batch size: {batch_size} (Available: {available_mb:.2f} MB, Sample: {sample_mb} MB)"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_to_s3_log(log, f"batch_size_adjust_{timestamp}.txt", config)
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
    save_to_s3_log(log, f"metrics_{timestamp}.txt", config)
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
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    s3_path = f"s3://{config['s3']['bucket']}/{config['s3']['outputs_path']}{config['s3']['subdirs']['plots']}{filename}.png"
    upload_to_s3(buf.getvalue(), s3_path, is_bytes=True)
    plt.close()

def save_to_s3_log(content, filename, config):
    s3_path = f"s3://{config['s3']['bucket']}/{config['s3']['outputs_path']}{config['s3']['subdirs']['logs']}{filename}"
    upload_to_s3(content.encode('utf-8'), s3_path, is_bytes=True)
    print(f"Saved log to {s3_path}")

if __name__ == "__main__":
    with open("../config/config.yaml", 'r') as f:
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