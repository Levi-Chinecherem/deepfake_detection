# Configuration Guide for Multimodal Deepfake Detection System

This `README.md` is your complete guide to setting up and understanding the `config/config.yaml` file for the Multimodal Deepfake Detection System. This file controls where data lives, how the system trains, what the models look like, and how resources are managed. Whether you’re training models, evaluating them, or testing videos, this config is the heart of it all. Let’s break it down step-by-step!

---

## What’s This File For?
The `config.yaml` file is a single place to define all settings for the system. It tells the scripts:
- Where to find data and save results in Amazon S3.
- How to train the models (e.g., how fast, how many rounds).
- What size the audio and images should be.
- How much computer memory to use.

It’s written in YAML—a simple format with key-value pairs—and is used by all scripts (`train.py`, `evaluate.py`, `test_videos.py`) to run the system on AWS EC2.

---

## Full Configuration File
Here’s the complete `config.yaml` with explanations for each part:

```yaml
# config/config.yaml

# S3 Settings
s3:
  bucket: "s3://your-bucket"  # Replace with your S3 bucket name
  dataset_path: "dataset/"    # Root path for training data in S3
  test_data_path: "test_data/" # Root path for video test data in S3
  outputs_path: "outputs/"    # Root path for results in S3
  subdirs:
    frames: "frames/"         # Images directory (training data)
    audio: "audio/"           # Audio directory (training data)
    models: "models/"         # Saved models
    plots: "plots/"           # Visualization plots
    logs: "logs/"             # Training logs
    results: "results/"       # Test results (CSV files)

# Hyperparameters
training:
  batch_size: 16              # Number of samples per batch (adjustable)
  learning_rate: 0.001        # How fast the model learns
  epochs: 10                  # Number of training rounds
  fusion_types:               # Which fusion strategies to use
    - "early"
    - "mid"
    - "late"
  dropout_rate: 0.3           # Prevent overfitting (30% dropout)

# Model Settings
models:
  audio:
    input_size: [128, 128]    # Spectrogram size (height, width)
    output_dim: 128           # Feature vector size
  image:
    input_size: [224, 224]    # Image size (height, width)
    output_dim: 128           # Feature vector size
  emotions:
    num_classes: 7            # Emotions: happy, sad, angry, etc.

# Memory Management
memory:
  max_usage_percent: 80       # Max memory usage (80%)
  target_usage_percent: 70    # Target usage for batch sizing (70%)

# Miscellaneous
logging:
  log_frequency: 10           # Log every 10 batches
```

---

## Setting It Up: Step-by-Step

### 1. Open the File
- On your EC2 instance or local machine, go to the `deepfake_detection/` folder:
  ```
  cd deepfake_detection/
  ```
- Open `config.yaml` with a text editor:
  ```
  nano config/config.yaml
  ```

### 2. Configure S3 Settings
The `s3` section tells the system where to find and store data in Amazon S3.

- **`bucket`**:
  - **What it is**: Your S3 bucket name (e.g., `s3://my-deepfake-bucket`).
  - **How to set it**: Replace `s3://your-bucket` with your actual bucket name. Create a bucket in AWS S3 if you don’t have one:
    - Go to AWS Console > S3 > Create Bucket.
    - Name it (e.g., `my-deepfake-bucket`).
    - Use the full path (e.g., `s3://my-deepfake-bucket`).
  - **Where it’s used**: All scripts (`train.py`, `evaluate.py`, `test_videos.py`) use this to access S3.

- **`dataset_path`**:
  - **What it is**: Root folder for training data (images and audio).
  - **Default**: `"dataset/"` (e.g., `s3://your-bucket/dataset/`).
  - **How to set it**: Keep as is unless your data is elsewhere (e.g., `data/training/`).
  - **Where it’s used**: `train.py` and `evaluate.py` for training and test splits.

- **`test_data_path`**:
  - **What it is**: Root folder for video files to test.
  - **Default**: `"test_data/"` (e.g., `s3://your-bucket/test_data/`).
  - **How to set it**: Keep as is; upload videos here for `test_videos.py`.
  - **Where it’s used**: `test_videos.py` to find videos for real-world testing.

- **`outputs_path`**:
  - **What it is**: Root folder for all results (models, logs, etc.).
  - **Default**: `"outputs/"` (e.g., `s3://your-bucket/outputs/`).
  - **How to set it**: Keep as is unless you want a different folder (e.g., `results/`).
  - **Where it’s used**: All scripts save outputs here.

- **`subdirs`**:
  - **What it is**: Subfolders under `outputs_path` and `dataset_path`.
  - **Settings**:
    - `frames: "frames/"`: Where training images live (e.g., `s3://your-bucket/dataset/frames/`).
    - `audio: "audio/"`: Where training audio lives (e.g., `s3://your-bucket/dataset/audio/`).
    - `models: "models/"`: Where trained models are saved (e.g., `.pth` files).
    - `plots: "plots/"`: Where visualization plots go (e.g., `.png` files).
    - `logs: "logs/"`: Where text logs go (e.g., `.txt` files).
    - `results: "results/"`: Where test CSV files go (e.g., `test_results_*.csv`).
  - **How to set it**: Keep defaults unless you reorganize S3 (e.g., change `logs` to `log_files/`).
  - **Where it’s used**: All scripts use these paths to read/write specific data.

### 3. Set Hyperparameters
The `training` section controls how the models learn.

- **`batch_size`**:
  - **What it is**: How many samples (audio/image pairs) to process at once.
  - **Default**: `16`.
  - **How to set it**: Lower to `8` or `4` if EC2 memory is low (e.g., 4GB); raise to `32` for more power.
  - **Where it’s used**: `train.py` and `evaluate.py` for data loading.

- **`learning_rate`**:
  - **What it is**: How fast the model updates its guesses (smaller = slower, more precise).
  - **Default**: `0.001`.
  - **How to set it**: Keep unless training is too slow (`0.01`) or unstable (`0.0001`).
  - **Where it’s used**: `train.py` for the Adam optimizer.

- **`epochs`**:
  - **What it is**: How many full passes through the training data.
  - **Default**: `10`.
  - **How to set it**: Lower to `5` for a quick test; raise to `20` for better accuracy.
  - **Where it’s used**: `train.py` for training loops.

- **`fusion_types`**:
  - **What it is**: Which model types to train/test (Early, Mid, Late Fusion).
  - **Default**: `["early", "mid", "late"]`.
  - **How to set it**: Remove a type (e.g., `["early", "mid"]`) to skip it.
  - **Where it’s used**: `train.py`, `evaluate.py`, `test_videos.py` to select models.

- **`dropout_rate`**:
  - **What it is**: Fraction of model connections to ignore to prevent overfitting.
  - **Default**: `0.3` (30%).
  - **How to set it**: Raise to `0.5` if overfitting; lower to `0.1` if underfitting.
  - **Where it’s used**: `models.py` in fusion model definitions.

### 4. Define Model Settings
The `models` section sets up audio and image processing.

- **`audio`**:
  - **`input_size: [128, 128]`**: Spectrogram size (height, width) for audio.
    - **How to set it**: Keep unless you change preprocessing (e.g., `[64, 64]` for smaller).
    - **Where it’s used**: `preprocess.py` and `models.py` for audio input.
  - **`output_dim: 128`**: Size of the audio feature vector.
    - **How to set it**: Keep unless you adjust model complexity (e.g., `64`).
    - **Where it’s used**: `models.py` for AudioModel output.

- **`image`**:
  - **`input_size: [224, 224]`**: Image size (height, width).
    - **How to set it**: Matches ResNet18 standard; adjust if using a different model (e.g., `[128, 128]`).
    - **Where it’s used**: `preprocess.py` and `models.py` for image input.
  - **`output_dim: 128`**: Size of the image feature vector.
    - **How to set it**: Keep unless you tweak fusion layers (e.g., `256`).
    - **Where it’s used**: `models.py` for ImageModel output.

- **`emotions`**:
  - **`num_classes: 7`**: Number of emotions (happy, sad, angry, fear, surprise, disgust, neutral).
    - **How to set it**: Keep unless you change emotion categories (e.g., `5`).
    - **Where it’s used**: `models.py` for emotion classification layers.

### 5. Manage Memory
The `memory` section keeps EC2 from crashing.

- **`max_usage_percent`**:
  - **What it is**: Maximum memory the system can use (stops if exceeded).
  - **Default**: `80` (80%).
  - **How to set it**: Lower to `70` for safety; raise to `90` if you have more RAM.
  - **Where it’s used**: `utils.py`, `train.py`, `evaluate.py`, `test_videos.py` for memory checks.

- **`target_usage_percent`**:
  - **What it is**: Target memory usage for batch sizing (stays below max).
  - **Default**: `70` (70%).
  - **How to set it**: Adjust with `max_usage_percent` (e.g., `60` if max is `70`).
  - **Where it’s used**: `utils.py` to dynamically set `batch_size`.

### 6. Set Logging Frequency
The `logging` section controls how often progress is saved.

- **`log_frequency`**:
  - **What it is**: How often to log batch details during training (every X batches).
  - **Default**: `10`.
  - **How to set it**: Raise to `20` for less logging; lower to `5` for more detail.
  - **Where it’s used**: `train.py` for batch logs.

### 7. Save and Test
- Save the file (Ctrl+O, Enter, Ctrl+X in `nano`).
- Test it loads correctly:
  ```
  python3 src/test_config.py --config config/config.yaml
  ```
- Look for: “Configuration loaded successfully!” with all settings printed.

---

## Where Each Part Is Used
- **`s3`**:
  - `train.py`: Reads `dataset_path`, saves to `models`, `logs`.
  - `evaluate.py`: Reads `dataset_path`, saves to `logs`, `plots`.
  - `test_videos.py`: Reads `test_data_path`, saves to `logs`, `results`.
- **`training`**:
  - `train.py`: Controls training process.
  - `evaluate.py`, `test_videos.py`: Uses `fusion_types` to select models.
- **`models`**:
  - `preprocess.py`: Shapes audio/images.
  - `models.py`: Defines model architecture.
- **`memory`**:
  - All scripts: Ensures memory safety.
- **`logging`**:
  - `train.py`: Logs training progress.

---

## Tips for Customization
- **Small EC2 Instance**: Lower `batch_size` to `8` and `epochs` to `5`.
- **Big Dataset**: Raise `epochs` to `20` and `batch_size` to `32` (if RAM allows).
- **Different Models**: Adjust `input_size` or `output_dim` in `models` section.
- **Less Logging**: Set `log_frequency` to `50`.

---

## Troubleshooting
- **S3 Paths Wrong**: Check bucket name and folder structure with `aws s3 ls s3://your-bucket/`.
- **Memory Errors**: Lower `batch_size` or `max_usage_percent`.
- **Config Not Loading**: Ensure YAML syntax (indentation, no tabs) is correct.

This config is your control panel—tweak it to fit your data and EC2 setup, and you’re ready to detect deepfakes!

---

### Notes
- **Detail**: Every key is explained with its purpose, default value, how to adjust it, and where it’s applied in the system.
- **User-Friendly**: Written for beginners, with clear steps and examples (e.g., AWS S3 setup).
- **Completeness**: Covers setup, usage, customization, and troubleshooting, tied to the latest `config.yaml` 