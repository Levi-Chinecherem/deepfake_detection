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
    """
    Monitor system memory usage and save log to S3.
    
    Args:
        config (dict): Configuration with S3 and memory settings.
        model_size_mb (int): Estimated model size in MB (default 100MB).
    
    Returns:
        float: Current memory usage percentage.
    """
    mem = psutil.virtual_memory()
    total_mb = mem.total / (1024 * 1024)  # Convert bytes to MB
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
    """
    Adjust batch size based on available memory (target 70% usage).
    
    Args:
        config (dict): Configuration with memory settings.
        sample_audio_mb (int): Estimated size of one audio sample in MB.
        sample_image_mb (int): Estimated size of one image sample in MB.
    
    Returns:
        int: Adjusted batch size.
    """
    mem = psutil.virtual_memory()
    total_mb = mem.total / (1024 * 1024)
    target_mb = total_mb * (config['memory']['target_usage_percent'] / 100)  # 70%
    available_mb = target_mb - 100  # Reserve 100MB for model
    
    sample_mb = sample_audio_mb + sample_image_mb  # Total per sample
    batch_size = max(1, int(available_mb / sample_mb))
    batch_size = min(batch_size, config['training']['batch_size'])  # Cap at config max
    
    log = f"Adjusted batch size: {batch_size} (Available: {available_mb:.2f} MB, Sample: {sample_mb} MB)"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_to_s3_log(log, f"batch_size_adjust_{timestamp}.txt", config)
    return batch_size

def calculate_metrics(y_true, y_pred, y_scores, config):
    """
    Compute evaluation metrics and save to S3.
    
    Args:
        y_true (list): True labels (0 or 1).
        y_pred (list): Predicted labels (0 or 1).
        y_scores (list): Prediction scores (probabilities).
        config (dict): Configuration for S3 path.
    
    Returns:
        dict: Metrics dictionary.
    """
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

def plot_and_save(plot_type, data, config, title, xlabel, ylabel, filename_prefix):
    """
    Generate and save a plot to S3 with legends and labels.
    
    Args:
        plot_type (str): Type of plot ('line', 'scatter', 'bar', 'matrix').
        data (dict): Data to plot (e.g., {'x': [], 'y': [], 'label': ''}).
        config (dict): Configuration for S3 path.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        filename_prefix (str): Prefix for unique filename.
    """
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
    
    # Save to in-memory buffer and upload to S3
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_path = (f"{config['s3']['bucket']}/{config['s3']['outputs_path']}"
               f"{config['s3']['subdirs']['plots']}{filename_prefix}_{timestamp}.png")
    upload_to_s3(buf.getvalue(), s3_path, is_bytes=True)
    plt.close()

def save_to_s3_log(content, filename, config):
    """
    Save text content to S3 as a log file.
    
    Args:
        content (str): Text to save.
        filename (str): Unique filename.
        config (dict): Configuration with S3 bucket and path.
    """
    s3_path = f"{config['s3']['bucket']}/{config['s3']['outputs_path']}{config['s3']['subdirs']['logs']}{filename}"
    upload_to_s3(content.encode('utf-8'), s3_path, is_bytes=True)
    print(f"Saved log to {s3_path}")

if __name__ == "__main__":
    # Load config
    with open("../config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Test memory monitoring
    percent = monitor_memory(config)
    print(f"Current memory usage: {percent:.2f}%")
    
    # Test batch size adjustment
    batch_size = adjust_batch_size(config)
    print(f"Adjusted batch size: {batch_size}")
    
    # Test metrics and plotting with dummy data
    y_true = [0, 1, 0, 1, 0]
    y_pred = [0, 1, 1, 1, 0]
    y_scores = [0.1, 0.9, 0.8, 0.7, 0.2]
    metrics, fpr, tpr, precision_curve, recall_curve = calculate_metrics(y_true, y_pred, y_scores, config)
    print("Metrics:", metrics)
    
    # Example plots
    plot_and_save('line', {'train': {'x': range(5), 'y': [0.5, 0.4, 0.3, 0.2, 0.1], 'label': 'Train Loss'}},
                  config, "Loss Curve", "Epoch", "Loss", "loss_curve")
    plot_and_save('scatter', {'roc': {'x': fpr, 'y': tpr, 'label': f"AUC = {metrics['roc_auc']:.2f}"}},
                  config, "ROC Curve", "False Positive Rate", "True Positive Rate", "roc_curve")
    plot_and_save('matrix', {'matrix': confusion_matrix(y_true, y_pred)},
                  config, "Confusion Matrix", "Predicted", "True", "confusion_matrix")