# Multimodal Deepfake Detection System

![Project Status](https://img.shields.io/badge/status-active-brightgreen)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/github/license/Levi-Chinecherem/deepfake_detection?color=orange)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/Levi-Chinecherem/deepfake_detection)

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Django](https://img.shields.io/badge/django-4.2.7-green?logo=django)
![PyTorch](https://img.shields.io/badge/pytorch-2.6.0-orange?logo=pytorch)
![Tailwind CSS](https://img.shields.io/badge/tailwindcss-3.4.1-blue?logo=tailwind-css)
![AWS](https://img.shields.io/badge/AWS-S3%20%26%20EC2-yellow?logo=amazon-aws)

![Contributors](https://img.shields.io/github/contributors/Levi-Chinecherem/deepfake_detection?color=green)
![Issues](https://img.shields.io/github/issues/Levi-Chinecherem/deepfake_detection?color=red)
![Pull Requests](https://img.shields.io/github/issues-pr/Levi-Chinecherem/deepfake_detection?color=purple)
![Stars](https://img.shields.io/github/stars/Levi-Chinecherem/deepfake_detection?style=social)
![Forks](https://img.shields.io/github/forks/Levi-Chinecherem/deepfake_detection?style=social)

![Code Size](https://img.shields.io/github/repo-size/Levi-Chinecherem/deepfake_detection)
![Research](https://img.shields.io/badge/Purpose-PhD%20Research-blue)

**A PhD Research Project by Bale Dennis**  
**Developed by Levi Chinecherem ([GitHub: Levi-Chinecherem](https://github.com/Levi-Chinecherem))**  

Welcome to the Multimodal Deepfake Detection System, a cutting-edge research initiative developed as part of Bale Dennis’s PhD dissertation. This system addresses the escalating challenge of deepfake detection by leveraging emotional consistency across audio and visual modalities, offering a novel contribution to digital forensics and machine learning. Built from the ground up and deployed on AWS infrastructure, this project combines academic rigor with practical application, making it a valuable tool for researchers, practitioners, and PhD evaluators alike.

---

## Research Objective
Deepfakes—synthetic media that convincingly mimic real content—threaten trust in digital communications. This PhD project, spearheaded by Bale Dennis, proposes a multimodal approach to detect deepfakes by analyzing the alignment of emotional cues between audio (e.g., voice intonation) and visuals (e.g., facial expressions). By integrating three distinct fusion strategies—Early, Mid, and Late Fusion—this system provides a robust framework to identify inconsistencies that betray synthetic manipulation, advancing the field of multimedia authentication.

---

## Key Innovations
- **Emotional Consistency Analysis**: Detects deepfakes by validating congruence between audio and visual emotional signals (e.g., a happy tone matching a smile).
- **Multimodal Fusion**: Implements three fusion models:
  - **Early Fusion**: Merges raw audio and visual data at the input stage.
  - **Mid Fusion**: Combines intermediate features for enhanced feature interaction.
  - **Late Fusion**: Processes modalities independently, aggregating predictions for balanced accuracy.
- **Cloud Scalability**: Operates on AWS EC2 with S3 integration, ensuring efficient data handling and scalability.
- **Resource Optimization**: Maintains memory usage below 80% with dynamic batch sizing, suitable for academic and real-world deployment.
- **Practical Testing**: Processes raw video files, producing CSV outputs for statistical analysis and comparison.

---

## Repository Structure
```
deepfake_detection/
├── config/
│   ├── config.yaml       # Configuration file (S3 paths, hyperparameters)
├── src/
│   ├── test_config.py    # Validates configuration loading
│   ├── s3_utils.py       # S3 streaming and storage utilities
│   ├── preprocess.py     # Preprocesses audio and image data
│   ├── models.py         # Defines Early, Mid, and Late Fusion models
│   ├── utils.py          # Memory management, metrics, and visualization tools
│   ├── train.py          # Trains the fusion models
│   ├── evaluate.py       # Evaluates model performance
│   ├── test_videos.py    # Tests real-world video files
├── requirements.txt      # Python dependencies
└── README.md             # This research documentation
```

---

## Setup and Deployment

### Prerequisites
- **AWS Account**: Required for EC2 and S3 access.
- **EC2 Instance**: Minimum `t2.medium` (2 vCPUs, 4GB RAM); GPU instances (e.g., `g4dn.xlarge`) optional for faster training.
- **S3 Bucket**: Stores datasets, models, and results.
- **Python 3.8+**: Core runtime environment.

### Installation Steps
1. **Clone the Repository**:
   ```
   git clone https://github.com/Levi-Chinecherem/deepfake_detection.git
   cd deepfake_detection/
   ```

2. **Configure EC2**:
   - Launch an EC2 instance (Amazon Linux 2 recommended).
   - Assign an IAM role with `AmazonS3FullAccess`.
   - Connect via SSH:
     ```
     ssh -i your-key.pem ec2-user@your-ec2-ip
     ```

3. **Deploy Code to EC2**:
   - Zip locally:
     ```
     zip -r deepfake_detection.zip deepfake_detection/
     ```
   - Upload to EC2:
     ```
     scp -i your-key.pem deepfake_detection.zip ec2-user@your-ec2-ip:/home/ec2-user/
     ```
   - Unzip on EC2:
     ```
     unzip deepfake_detection.zip
     cd deepfake_detection/
     ```

4. **Install Dependencies**:
   - Update EC2 and install Python:
     ```
     sudo yum update -y
     sudo yum install python3 -y
     ```
   - Install `pip` and libraries:
     ```
     curl -O https://bootstrap.pypa.io/get-pip.py
     python3 get-pip.py --user
     pip3 install -r requirements.txt
     ```
   - Install FFmpeg for video processing:
     ```
     sudo yum install ffmpeg -y
     ```

### Configuration
1. **Edit `config.yaml`**:
   - Open:
     ```
     nano config/config.yaml
     ```
   - Update `s3.bucket` to your bucket (e.g., `s3://bale-dennis-phd-bucket`).
   - Customize hyperparameters (e.g., `batch_size`, `epochs`) as needed.
   - Save and exit (Ctrl+O, Enter, Ctrl+X).

2. **Prepare S3**:
   - Create your bucket:
     ```
     aws s3 mb s3://bale-dennis-phd-bucket
     ```
   - Set up directories:
     ```
     aws s3 mkdir s3://bale-dennis-phd-bucket/dataset/
     aws s3 mkdir s3://bale-dennis-phd-bucket/test_data/
     aws s3 mkdir s3://bale-dennis-phd-bucket/outputs/
     ```

---

## Usage Instructions

### 1. Model Training
- **Objective**: Train the fusion models on a labeled dataset.
- **Command**:
  ```
  python3 src/train.py --config config/config.yaml
  ```
- **Input**: Upload images to `s3://bale-dennis-phd-bucket/dataset/frames/` and audio to `s3://bale-dennis-phd-bucket/dataset/audio/` (subdirs: `train/`, `test/`, `validate/` with `real/` and `fake/`).
- **Output**: Models in `outputs/models/` (e.g., `mid_epoch_5_*.pth`), logs in `outputs/logs/`.

### 2. Model Evaluation
- **Objective**: Assess model performance on the test split.
- **Command**:
  ```
  python3 src/evaluate.py --config config/config.yaml
  ```
- **Output**: Metrics (e.g., accuracy, F1) in `outputs/logs/`, visualization plots (e.g., ROC curves) in `outputs/plots/`.

### 3. Real-World Video Testing
- **Objective**: Detect deepfakes in unlabeled video files, a key demonstration for Bale Dennis’s PhD defense.
- **Command**:
  ```
  python3 src/test_videos.py --config config/config.yaml
  ```
- **Input**: Upload videos (e.g., `.mp4`, `.avi`) to `s3://bale-dennis-phd-bucket/test_data/`.
- **Output**: CSV results in `outputs/results/` (e.g., `test_results_20250302_123456.csv`), memory logs in `outputs/logs/`.
- **CSV Format**:
  ```csv
  video_name,early_pred,mid_pred,late_pred,avg_pred,is_fake,mid_consistency,late_consistency,processing_time,ground_truth,system_a_pred,system_b_pred
  video1.mp4,0.35,0.45,0.40,0.40,0,0.85,0.90,2.34,,,
  ```

---

## Research Outputs
- **Training Phase**: Model weights and training logs for methodological validation.
- **Evaluation Phase**: Comprehensive metrics (e.g., AUC-ROC, F1-score) and visualizations for performance assessment.
- **Testing Phase**: CSV files with prediction probabilities and emotional consistency scores, facilitating statistical analysis and comparison with existing methods.

---

## Customization for Research
- **Hyperparameters**: Adjust `config.yaml`:
  - `batch_size`: Lower to `8` for limited resources; increase to `32` for faster training.
  - `epochs`: Set to `20` for deeper learning; `5` for quick experiments.
  - `fusion_types`: Test specific models (e.g., `["mid", "late"]`).
- **Data**: Use custom datasets in `dataset/` or videos in `test_data/` to replicate or extend Bale Dennis’s experiments.

---

## Troubleshooting
- **S3 Access Denied**: Confirm IAM role permissions and bucket name accuracy.
- **Memory Exceeded**: Reduce `batch_size` or adjust `memory.max_usage_percent` in `config.yaml`.
- **Model Not Found**: Ensure `train.py` completed; check `outputs/models/` for `.pth` files.
- **Video Processing Failure**: Verify FFmpeg installation (`ffmpeg -version`) and video format compatibility.

---

## Academic Significance
Developed under Bale Dennis’s PhD research, this system contributes to the field by:
- Proposing emotional consistency as a novel deepfake detection criterion.
- Demonstrating multimodal fusion’s efficacy in multimedia forensics.
- Providing a reproducible, cloud-based framework for future studies.

This repository serves as a cornerstone for Bale Dennis’s dissertation defense, offering a practical implementation of his theoretical advancements.

---

## Contributing
Contributions are encouraged to enhance this research tool! Fork the repo, report issues, or submit pull requests with improvements such as:
- Enhanced feature extraction techniques.
- Additional fusion methodologies.
- Integration with deepfake benchmark datasets.

---

## License
Licensed under the [MIT License](LICENSE), this project is freely available for academic research and educational purposes.

---

## Credits
- **Bale Dennis**: PhD candidate and principal investigator, whose research vision shaped this system.
- **Levi Chinecherem**: Developer and technical architect ([GitHub: Levi-Chinecherem](https://github.com/Levi-Chinecherem)), responsible for implementation.

For inquiries, contact Bale Dennis via his academic institution or Levi Chinecherem through GitHub Issues. Together let's send Best wishes to Bale Dennis for a successful PhD defense!
