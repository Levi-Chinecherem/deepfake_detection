# src/evaluate.py
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from preprocess import DeepfakeDataset
from models import EarlyFusionModel, MidFusionModel, LateFusionModel, emotional_consistency
from utils import monitor_memory, adjust_batch_size, calculate_metrics, plot_and_save, save_to_s3_log
from sklearn.metrics import confusion_matrix
from s3_utils import stream_from_s3, list_s3_files

def evaluate_model(model, dataloader, config, fusion_type, device):
    model.eval()
    y_true, y_pred, y_scores, consistency_scores = [], [], [], []
    
    with torch.no_grad():
        for audio, image, label in dataloader:
            audio, image, label = audio.to(device), image.to(device), label.to(device).view(-1, 1)
            
            if fusion_type == 'early':
                pred, _ = model(audio, image)
            elif fusion_type == 'mid':
                pred, _, audio_em, image_em = model(audio, image)
                consistency = emotional_consistency(audio_em, image_em)
                consistency_scores.extend(consistency.cpu().numpy())
            else:  # late
                pred, audio_em, image_em = model(audio, image)
                consistency = emotional_consistency(audio_em, image_em)
                consistency_scores.extend(consistency.cpu().numpy())
            
            y_true.extend(label.cpu().numpy().flatten())
            y_pred.extend((pred > 0.5).float().cpu().numpy().flatten())
            y_scores.extend(pred.cpu().numpy().flatten())
            
            mem_percent = monitor_memory(config)
            if mem_percent > config['memory']['max_usage_percent']:
                raise Exception(f"Stopping: Memory usage ({mem_percent}%) exceeds limit")
    
    return y_true, y_pred, y_scores, consistency_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection models.")
    parser.add_argument("--config", type=str, default="../config/config.yaml",
                        help="Path to config YAML file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_dataset = DeepfakeDataset(config, split="test")
    batch_size = adjust_batch_size(config)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
    
    for fusion_type in config['training']['fusion_types']:
        model_path = sorted(list_s3_files(f"s3://{config['s3']['bucket']}/{config['s3']['outputs_path']}{config['s3']['subdirs']['models']}"), reverse=True)
        latest_model = next((p for p in model_path if fusion_type in p), None)
        if latest_model:
            print(f"Loading {fusion_type} model from {latest_model}")
            model_stream = stream_from_s3(latest_model)
            state_dict = torch.load(model_stream, map_location=device)
            models_dict[fusion_type].load_state_dict(state_dict)
        else:
            print(f"No trained {fusion_type} model found in S3")
            continue
        
        y_true, y_pred, y_scores, consistency_scores = evaluate_model(models_dict[fusion_type], test_loader,
                                                                     config, fusion_type, device)
        
        metrics, fpr, tpr, precision_curve, recall_curve = calculate_metrics(y_true, y_pred, y_scores, config)
        print(f"{fusion_type.capitalize()} Fusion Metrics:", metrics)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{fusion_type}"
        
        plot_and_save('scatter', {'roc': {'x': fpr, 'y': tpr, 'label': f"AUC = {metrics['roc_auc']:.2f}"}},
                      config, "ROC Curve", "False Positive Rate", "True Positive Rate", f"{prefix}_roc_curve_{timestamp}")
        
        plot_and_save('scatter', {'pr': {'x': recall_curve, 'y': precision_curve, 'label': f"AUC = {metrics['pr_auc']:.2f}"}},
                      config, "Precision-Recall Curve", "Recall", "Precision", f"{prefix}_pr_curve_{timestamp}")
        
        plot_and_save('matrix', {'matrix': confusion_matrix(y_true, y_pred)},
                      config, "Confusion Matrix", "Predicted", "True", f"{prefix}_confusion_matrix_{timestamp}")
        
        plot_and_save('bar', {'dist': {'x': range(len(y_scores)), 'y': y_scores, 'label': 'Prediction Scores'}},
                      config, "Prediction Distribution", "Sample", "Score", f"{prefix}_pred_dist_{timestamp}")
        
        if fusion_type != 'early' and consistency_scores:
            plot_and_save('bar', {'consistency': {'x': range(len(consistency_scores)), 'y': consistency_scores, 'label': 'Consistency'}},
                          config, "Emotional Consistency Histogram", "Sample", "Cosine Similarity", f"{prefix}_consistency_hist_{timestamp}")

if __name__ == "__main__":
    main()