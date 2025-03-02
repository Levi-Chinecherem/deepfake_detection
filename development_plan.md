### Development Plan: Step-by-Step

#### Step 1: Configuration Setup
- **Purpose**: Create a configuration file to store all settings (S3 bucket, hyperparameters, EC2 specs) in one place, making the system flexible and easy to adjust.
- **Functionalities**:
  - Define S3 bucket name and paths (`dataset/`, `outputs/`).
  - Set hyperparameters: batch size, learning rate, epochs, fusion types.
  - Specify EC2 memory limits (80% max) and model paths.
  - Use YAML format for readability and parsing.
- **Tools**:
  - Python: Base language.
  - `pyyaml`: Library to read/write YAML files (`pip install pyyaml`).
- **Details**:
  - File: `config/config.yaml`.
  - Example settings: S3 bucket (`s3://my-deepfake-bucket`), batch size (16), learning rate (0.001).
  - Command-line argument to load config in scripts (e.g., `--config config.yaml`).
- **Output**: A single `config.yaml` file in `config/` folder.

#### Step 2: S3 Utilities
- **Purpose**: Build tools to stream data from S3, create folders, and save results without local downloads.
- **Functionalities**:
  - Stream audio/image files from S3 using `boto3` and `io.BytesIO`.
  - Check/create S3 folders (`outputs/models/`, `outputs/plots/`, `outputs/logs/`).
  - Upload files (models, plots, logs) to S3.
  - Handle errors (e.g., missing bucket, permissions).
- **Tools**:
  - `boto3`: AWS SDK for Python (`pip install boto3`).
  - `io`: In-memory file handling.
  - Python: Core logic.
- **Details**:
  - File: `src/s3_utils.py`.
  - Functions: `stream_from_s3()`, `create_s3_folder()`, `upload_to_s3()`.
  - Uses config from Step 1 for bucket name.
  - Streams data in chunks to manage memory.
- **Output**: A reusable `s3_utils.py` module.

#### Step 3: Data Preprocessing
- **Purpose**: Prepare audio and image data in-memory from S3 streams for model training.
- **Functionalities**:
  - Audio: Stream WAV, convert to spectrograms (128x128), normalize.
  - Images: Stream JPG/PNG, resize to 224x224, normalize to [0, 1].
  - Create PyTorch datasets to load data in batches.
  - Pair audio/image randomly (since not from same source) with real/fake labels.
- **Tools**:
  - `librosa`: Audio processing (`pip install librosa`).
  - `PIL` (Pillow): Image processing (`pip install Pillow`).
  - `torch`: PyTorch for tensors (`pip install torch`).
  - `torchvision`: Image transforms (`pip install torchvision`).
  - `boto3`: Via `s3_utils.py`.
- **Details**:
  - File: `src/preprocess.py`.
  - Functions: `preprocess_audio()`, `preprocess_image()`, `DeepfakeDataset` class.
  - Batch size from config; memory-checked later.
- **Output**: Preprocessed data ready for models.

#### Step 4: Model Definitions
- **Purpose**: Define the neural network models for audio, image, and fusion strategies.
- **Functionalities**:
  - Audio Model: CNN + LSTM for emotion features (128-dim output).
  - Image Model: ResNet18 for emotion features (128-dim output).
  - Early Fusion: Concatenate preprocessed inputs, joint CNN.
  - Mid Fusion: Concatenate feature vectors, dense layers.
  - Late Fusion: Separate models, combine predictions.
  - Emotional consistency check via cosine similarity.
- **Tools**:
  - `torch`: PyTorch for model building.
  - `torch.nn`: Neural network layers.
  - `torchvision.models`: Pretrained ResNet18.
- **Details**:
  - File: `src/models.py`.
  - Classes: `AudioModel`, `ImageModel`, `EarlyFusionModel`, `MidFusionModel`, `LateFusionModel`.
  - Activation: ReLU (CNN), Tanh (LSTM), Sigmoid (output).
  - Save emotion vectors for consistency check.
- **Output**: Model classes ready for training.

#### Step 5: Utility Functions
- **Purpose**: Add helper functions for memory management, metrics, and plotting.
- **Functionalities**:
  - Monitor memory usage (cap at 80%) with `psutil`.
  - Calculate dynamic batch size based on memory.
  - Compute metrics: accuracy, precision, recall, F1, AUC-ROC, etc.
  - Generate plots (loss, accuracy, confusion matrix, etc.) with legends/labels.
  - Save results to S3 via `s3_utils.py`.
- **Tools**:
  - `psutil`: Memory monitoring (`pip install psutil`).
  - `matplotlib`: Plotting (`pip install matplotlib`).
  - `sklearn`: Metrics (`pip install scikit-learn`).
  - `torch`: Tensor operations.
- **Details**:
  - File: `src/utils.py`.
  - Functions: `monitor_memory()`, `adjust_batch_size()`, `calculate_metrics()`, `plot_and_save()`.
  - All plots saved as PNGs to S3.
- **Output**: Reusable utilities for training/evaluation.

#### Step 6: Training Script
- **Purpose**: Train the models on EC2 with data from S3, saving everything to S3.
- **Functionalities**:
  - Load config and initialize models.
  - Stream data in batches, preprocess, and train (early/mid/late fusion).
  - Monitor memory, adjust batch size if >70% (target 80% max).
  - Save intermediate results (emotion vectors, losses) and models to S3.
  - Log progress (epoch, loss, accuracy) to S3.
- **Tools**:
  - `torch`: Training loop, optimizers (Adam).
  - `argparse`: Command-line args (`--config`).
  - All modules from previous steps.
- **Details**:
  - File: `src/train.py`.
  - Command: `python src/train.py --config config.yaml`.
  - Loss: Binary cross-entropy + consistency loss.
  - Saves every epoch to S3.
- **Output**: Trained models and logs in S3.

#### Step 7: Evaluation Script
- **Purpose**: Evaluate the trained models and generate all visualizations.
- **Functionalities**:
  - Load models from S3.
  - Stream test data, predict real/fake.
  - Calculate all metrics (accuracy, F1, AUC, etc.).
  - Generate 19 plots (loss curves, ROC, confusion matrix, etc.).
  - Save everything to S3.
- **Tools**:
  - `torch`: Model inference.
  - `sklearn`: Metrics.
  - `matplotlib`: Plots.
  - All previous modules.
- **Details**:
  - File: `src/evaluate.py`.
  - Command: `python src/evaluate.py --config config.yaml`.
  - Plots include legends, labels, saved as PNGs.
- **Output**: Metrics and plots in S3.

#### Step 8: Final Integration and Testing
- **Purpose**: Tie everything together and test on EC2.
- **Functionalities**:
  - Verify all scripts work end-to-end (train → evaluate).
  - Test memory usage stays <80%.
  - Confirm all outputs (models, plots, logs, intermediates) reach S3.
  - Debug any S3 streaming or model issues.
- **Tools**:
  - All previous tools.
  - EC2 terminal for execution.
- **Details**:
  - Run full pipeline: `train.py` then `evaluate.py`.
  - Check S3 bucket for completeness.
- **Output**: Fully functional system.

---

### Development Workflow
1. **Start with Step 1**: Build the config file to set the foundation.
2. **Progress Sequentially**: Each step builds on the previous one.
3. **Request Code**: Ask for any step’s full code when ready (e.g., “Give me Step 2 code”).
4. **Test on EC2**: Deploy and run on EC2 after coding locally or directly there.

---

### Notes to Minimize Errors
- **Dependencies**: Listed in `requirements.txt` (created in Step 1).
  ```
  pyyaml
  boto3
  librosa
  Pillow
  torch
  torchvision
  psutil
  matplotlib
  scikit-learn
  ```
- **Error Handling**: Each step includes try-except blocks for S3 access, memory limits, etc.
- **Modularity**: Self-contained files reduce interdependence issues.
- **Testing**: I’ll simulate S3/EC2 constraints to ensure code works as expected.
