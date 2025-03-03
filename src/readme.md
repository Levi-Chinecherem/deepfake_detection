
# Multimodal Deepfake Detection System - Video Testing Guide

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

This guide explains how to test the Multimodal Deepfake Detection System on real-world video files. The system uses trained models to analyze videos, checking emotional consistency between audio and visuals to detect deepfakes. You’ll upload videos to Amazon S3, run a testing script on AWS EC2, and get results in a CSV file for analysis. Let’s get started!

---

## What Does This Testing Do?
The testing script (`test_videos.py`) takes video files (e.g., `.mp4`, `.avi`) from an S3 folder, processes them with three trained models (Early, Mid, and Late Fusion), and predicts if they’re real or fake. It:
- Extracts audio and a frame from each video.
- Runs them through the models to get prediction scores and emotional consistency.
- Saves results in a CSV file to S3, ready for statistical analysis or comparison with other systems.

---

## Folder Structure
### On Your EC2 Instance
```
deepfake_detection/
├── config/
│   ├── config.yaml       # Settings (S3 bucket, model details)
├── src/
│   ├── s3_utils.py       # S3 streaming and saving
│   ├── preprocess.py     # Audio/image preprocessing
│   ├── models.py         # Trained model definitions
│   ├── utils.py          # Memory and saving utilities
│   ├── test_videos.py    # Video testing script
├── requirements.txt      # Python dependencies
└── README.md             # This guide
```
*Note*: Assumes training (`train.py`) and evaluation (`evaluate.py`) scripts exist but aren’t needed for testing.

### In S3
```
s3://your-bucket/
├── test_data/            # Upload your video files here
│   ├── video1.mp4
│   ├── video2.avi
│   └── ...
├── outputs/
│   ├── models/          # Trained models (from training)
│   │   ├── early_epoch_10_20250302_123456.pth
│   │   ├── mid_epoch_10_20250302_123456.pth
│   │   └── late_epoch_10_20250302_123456.pth
│   ├── logs/            # Memory logs
│   └── results/         # CSV results (created automatically)
```

---

## How to Test Videos on EC2

### 1. Set Up Your EC2 Instance
- **Instance**: Use a `t2.medium` (2 vCPUs, 4GB RAM) or better.
- **IAM Role**: Attach S3 full access (e.g., `AmazonS3FullAccess`).
- **Connect**: SSH into EC2:
  ```
  ssh -i your-key.pem ec2-user@your-ec2-ip
  ```

### 2. Upload the Code
- Assuming the system is built, zip the `deepfake_detection/` folder locally:
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

### 3. Install Dependencies
- Update EC2 and install Python:
  ```
  sudo yum update -y
  sudo yum install python3 -y
  ```
- Install `pip` and Python libraries:
  ```
  curl -O https://bootstrap.pypa.io/get-pip.py
  python3 get-pip.py --user
  pip3 install -r requirements.txt
  ```
- Install FFmpeg for video processing:
  ```
  sudo yum install ffmpeg -y
  ```

### 4. Configure Your S3 Bucket
- Edit `config/config.yaml`:
  ```
  nano config/config.yaml
  ```
- Ensure the S3 bucket matches yours (e.g., `s3://my-deepfake-bucket`):
  ```yaml
  s3:
    bucket: "s3://my-deepfake-bucket"
    outputs_path: "outputs/"
    subdirs:
      models: "models/"
      logs: "logs/"
      results: "results/"  # Add this if not present
  ```
- Save and exit (Ctrl+O, Enter, Ctrl+X).

### 5. Upload Test Videos
- Create the `test_data/` folder in S3:
  ```
  aws s3 mkdir s3://your-bucket/test_data/
  ```
- Upload your video files (e.g., `.mp4`, `.avi`):
  ```
  aws s3 cp video1.mp4 s3://your-bucket/test_data/
  aws s3 cp video2.avi s3://your-bucket/test_data/
  ```

### 6. Run the Video Testing
- Execute the script:
  ```
  python3 src/test_videos.py --config config/config.yaml
  ```
- **What Happens**:
  - Streams videos from `s3://your-bucket/test_data/`.
  - Loads the latest trained models from `s3://your-bucket/outputs/models/`.
  - Extracts audio and a frame from each video.
  - Predicts deepfake probabilities and emotional consistency.
  - Saves a CSV to `s3://your-bucket/outputs/results/` (e.g., `test_results_20250302_123456.csv`).
  - Logs memory usage to `s3://your-bucket/outputs/logs/`.
  - Keeps memory below 80%.

- **Sample Output** (Console):
  ```
  Using device: cpu
  Loading early model from s3://your-bucket/outputs/models/early_epoch_10_20250302_123456.pth
  Loading mid model from s3://your-bucket/outputs/models/mid_epoch_10_20250302_123456.pth
  Loading late model from s3://your-bucket/outputs/models/late_epoch_10_20250302_123456.pth
  Processing s3://your-bucket/test_data/video1.mp4
  Saved log to s3://your-bucket/outputs/logs/memory_usage_20250302_123456.txt
  Processing s3://your-bucket/test_data/video2.avi
  Saved results to s3://your-bucket/outputs/results/test_results_20250302_123456.csv
  ```

---

## Checking the Results
- **CSV File**: Download from S3:
  ```
  aws s3 cp s3://your-bucket/outputs/results/test_results_20250302_123456.csv .
  ```
- **Sample CSV**:
  ```csv
  video_name,early_pred,mid_pred,late_pred,avg_pred,is_fake,mid_consistency,late_consistency,processing_time,ground_truth,system_a_pred,system_b_pred
  video1.mp4,0.35,0.45,0.40,0.40,0,0.85,0.90,2.34,,,
  video2.avi,0.75,0.80,0.78,0.78,1,0.45,0.50,2.56,,,
  ```
  - **Columns**:
    - `video_name`: Video filename.
    - `early_pred`, `mid_pred`, `late_pred`: Model probabilities (0-1).
    - `avg_pred`: Average of predictions.
    - `is_fake`: Binary result (0 = real, 1 = fake; threshold 0.5).
    - `mid_consistency`, `late_consistency`: Emotional consistency scores (cosine similarity).
    - `processing_time`: Time taken (seconds).
    - `ground_truth`, `system_a_pred`, `system_b_pred`: Optional for comparison (blank here).

- **Logs**: Memory usage logs in `s3://your-bucket/outputs/logs/` (e.g., `memory_usage_20250302_123456.txt`).

---

## Troubleshooting
- **S3 Errors**: Check bucket name in `config.yaml` and IAM permissions.
- **Missing Models**: Ensure training completed (`outputs/models/` has `.pth` files).
- **Memory Issues**: If >80%, reduce video size or process fewer at once (edit script to limit files).
- **FFmpeg Errors**: Verify FFmpeg is installed (`ffmpeg -version`).

---

## Using the Results
The CSV is designed for statistical analysis:
- Compare `avg_pred` and `is_fake` with `ground_truth` (if added) for accuracy.
- Analyze `mid_consistency` and `late_consistency` for emotional mismatch trends.
- Add `system_a_pred` and `system_b_pred` manually to compare with other deepfake detectors.

Upload new videos to `test_data/` and re-run anytime to test more!

---

### Notes
- **Focus**: This `README.md` is tailored to testing with `test_videos.py`, omitting training/evaluation details for clarity.
- **CSV Format**: Matches your request, with flexibility for future comparisons
- **Real-World Use**: Assumes trained models exist; processes one video at a time to mimic practical deployment.
- **S3 Structure**: Adds `test_data/` and `outputs/results/`—update `config.yaml` if `results/` isn’t in your original subdirs.
