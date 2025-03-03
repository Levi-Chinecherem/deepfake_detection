# Deepfake Detection Web Interface

**A PhD Research Project by Bale Dennis**  
**Developed by Levi Chinecherem ([GitHub: Levi-Chinecherem](https://github.com/Levi-Chinecherem))**  

Welcome to the Deepfake Detection Web Interface, a standalone web application developed as part of Bale Dennis’s PhD research. This project provides an intuitive, user-friendly dashboard for detecting deepfakes in video files by leveraging pretrained multimodal machine learning models. Built with Django and styled using Tailwind CSS, it allows users to upload videos via a web browser, processes them using advanced fusion techniques, and displays results with emotional consistency analysis. Deployed on AWS EC2 with Amazon S3 integration, this interface showcases the practical application of Bale Dennis’s research for his PhD defense.

---

## Project Overview
This web interface enables users to upload videos and detect deepfakes by analyzing emotional consistency between audio and visual modalities, a core innovation from Bale Dennis’s PhD work. It uses three pretrained fusion models:
- **Early Fusion**: Combines audio and visuals at the input stage.
- **Mid Fusion**: Merges intermediate features for joint analysis.
- **Late Fusion**: Averages separate predictions for balanced outcomes.

Results are presented with a clear “Real” or “Fake” verdict, detailed prediction scores, and processing logs, stored securely in S3.

---

## Features
- **Video Upload**: Upload videos via a sleek web interface.
- **Deepfake Detection**: Processes videos with pretrained models (Early, Mid, Late Fusion).
- **Emotional Consistency**: Displays cosine similarity scores for audio-visual alignment.
- **Responsive Design**: Modern UI with Tailwind CSS, featuring a gold light theme and grayish-blue dark theme.
- **Real-Time Results**: Shows detection outcomes and logs instantly after processing.
- **S3 Integration**: Securely stores uploads, logs, and results in Amazon S3 using `.env` credentials.
- **Theme Toggle**: Switch between light (gold) and dark (gray-blue) modes with persistence.

---

## Repository Structure
```
web_interface/
├── .env                   # S3 credentials (not tracked)
├── config.yaml            # Web app configuration
├── deepfake_web/          # Django project folder
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── detection_app/         # Django app folder
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   ├── ml_models.py       # Pretrained model definitions
│   ├── preprocess.py      # Audio/image preprocessing
│   ├── s3_utils.py        # S3 utilities
│   ├── views.py           # Backend logic
│   ├── templates/
│   │   └── detection_app/
│   │       └── index.html # Frontend UI
│   └── urls.py
├── static/
│   └── loading.gif        # Loading animation
├── manage.py              # Django management script
└── requirements.txt       # Dependencies
```

---

## Prerequisites
- **AWS Account**: For EC2 deployment and S3 storage.
- **EC2 Instance**: Minimum `t2.medium` (2 vCPUs, 4GB RAM).
- **S3 Bucket**: To store user data, logs, results, and pretrained models.
- **Python 3.8+**: For running the Django app.

## Setup Instructions

### 1. Clone the Repository
Clone this standalone web interface:
```bash
git clone https://github.com/Levi-Chinecherem/deepfake_detection.git
cd deepfake_detection/web_interface/
```

### 2. Set Up EC2
- Launch an EC2 instance (e.g., Amazon Linux 2).
- Connect via SSH:
  ```bash
  ssh -i your-key.pem ec2-user@your-ec2-ip
  ```
- Update and install Python:
  ```bash
  sudo yum update -y
  sudo yum install python3 -y
  ```

### 3. Install Dependencies
- Install `pip` and project dependencies:
  ```bash
  curl -O https://bootstrap.pypa.io/get-pip.py
  python3 get-pip.py --user
  pip3 install -r requirements.txt
  ```
- Install FFmpeg for video processing:
  ```bash
  sudo yum install ffmpeg -y
  ```

### 4. Configure Environment
- Create `.env` in `web_interface/`:
  ```bash
  nano .env
  ```
  Add your S3 credentials:
  ```
  AWS_ACCESS_KEY_ID=your_access_key_id
  AWS_SECRET_ACCESS_KEY=your_secret_access_key
  AWS_REGION=us-east-1
  S3_BUCKET=your-bucket-name
  ```
  - Replace values with your AWS IAM credentials, region, and bucket name.
  - Save and exit (Ctrl+O, Enter, Ctrl+X).

### 5. Prepare S3
- Create required S3 folders:
  ```bash
  aws s3 mkdir s3://your-bucket-name/users/data/
  aws s3 mkdir s3://your-bucket-name/users/logs/
  aws s3 mkdir s3://your-bucket-name/users/results/
  aws s3 mkdir s3://your-bucket-name/users/models/
  ```
- Copy pretrained models (e.g., `early_epoch_10_*.pth`) to `users/models/`:
  ```bash
  aws s3 cp path/to/models/ s3://your-bucket-name/users/models/ --recursive
  ```

### 6. Run the Server
- Start the Django development server:
  ```bash
  python manage.py runserver 0.0.0.0:8000
  ```
- Access at `http://your-ec2-ip:8000`.

---

## Usage
1. **Open the Dashboard**:
   - Visit `http://your-ec2-ip:8000` in a browser.

2. **Upload a Video**:
   - Click “Choose File” and select a video (e.g., `.mp4`).
   - Click “Upload” to process it with pretrained models.

3. **View Results**:
   - See the verdict (“Real” or “Fake”), prediction scores, and consistency metrics.
   - Check processing logs below the results.

4. **Toggle Theme**:
   - Click the sun/moon icon to switch between light (gold) and dark (gray-blue) modes.

---

## Results
- **S3 Storage**:
  - Uploaded videos: `s3://your-bucket-name/users/data/`
  - Logs: `s3://your-bucket-name/users/logs/`
  - Results: `s3://your-bucket-name/users/results/`
- **Output Format**: CSV files with fields like `video_name`, `early_pred`, `mid_pred`, `late_pred`, `avg_pred`, `is_fake`, `mid_consistency`, `late_consistency`.

---

## Customization
- **Theme Colors**:
  - Edit `index.html` under `<style>` and Tailwind config to adjust `--gold-primary`, `--dark-milky`, etc.
- **S3 Paths**:
  - Modify `config.yaml` to change `users.data`, `users.logs`, etc.
- **Model Settings**:
  - Update `config.yaml` under `models` for different input sizes.

---

## Troubleshooting
- **Server Won’t Start**:
  - Check `.env` for correct S3 credentials.
  - Ensure `config.yaml` and pretrained models are in place.
- **Upload Fails**:
  - Verify FFmpeg (`ffmpeg -version`) and S3 permissions.
- **Theme Issues**:
  - Clear browser cache if toggle doesn’t update colors.

---

## Academic Context
This web interface, developed by Levi Chinecherem for Bale Dennis’s PhD defense, demonstrates the practical deployment of multimodal deepfake detection using emotional consistency. It’s a standalone showcase of Bale’s research, requiring only pretrained models from the original system.

---

## Contributing
Contributions welcome! Fork the repo, submit issues, or pull requests to enhance:
- Real-time processing feedback.
- Additional UI features (e.g., progress bars).
- Model visualization tools.

---

## License
Licensed under the [MIT License](LICENSE), free for academic and research use.

---

## Credits
- **Bale Dennis**: PhD candidate and research lead.
- **Levi Chinecherem**: Developer ([GitHub: Levi-Chinecherem](https://github.com/Levi-Chinecherem)).

For inquiries, contact Bale Dennis via academic channels or Levi Chinecherem through GitHub Issues. As usual Best wishes for the PhD defense!
