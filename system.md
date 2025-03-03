# Multimodal Deepfake Detection System

Welcome to the Multimodal Deepfake Detection System! This project builds a machine learning system to detect deepfakes—videos or media that have been manipulated to look or sound real but aren’t. Our system uses two types of data: **images** (like snapshots of faces) and **audio** (like voice recordings) to figure out if something is real or fake. It does this by checking if the emotions in the face (like a smile) match the emotions in the voice (like a happy tone). If they don’t match, it’s more likely a fake!

We’re building this from scratch, training it on data stored in **Amazon S3** (a cloud storage service), and running it on **AWS EC2** (a cloud computer). Everything—data, results, and pictures—stays in the cloud, and we’ll make sure it doesn’t use too much memory so the computer doesn’t crash. Let’s break it down step-by-step so you can understand exactly what’s happening.

---

## What Are We Building?
Imagine you’re watching a video of someone talking. How can you tell if it’s real or a clever fake? Our system looks at two clues:
1. **Images**: Pictures of the person’s face (one at a time).
2. **Audio**: The sound of their voice.

It checks if the emotions in the face (like looking angry) match the emotions in the voice (like sounding angry). Real videos usually have matching emotions, but fakes often don’t because it’s hard to fake both perfectly. The system learns this by studying lots of examples and then predicts: **Real** or **Fake**.

We use three ways to combine the clues (**fusion**):
- **Early Fusion**: Mix the image and audio data right at the start and analyze them together.
- **Mid Fusion**: Look at the image and audio separately first, then mix them halfway through.
- **Late Fusion**: Look at them completely separately and only mix the final guesses.

Everything runs on a cloud computer (EC2), and all the data lives in S3 buckets (like cloud folders). We don’t download anything to a local computer—we stream it straight from the cloud!

---

## Why This System?
Deepfakes are becoming more common, and they can trick people—like making a fake video of a politician saying something they didn’t. Our system helps spot these fakes by focusing on emotional consistency, which is hard for deepfake creators to get right. It’s lightweight (doesn’t need a super powerful computer) and saves everything to the cloud so you don’t lose any details.

---

## How Does It Work?
Here’s the big picture of how we build and run this system:

### 1. Where’s the Data?
Our data is stored in **Amazon S3**, a cloud storage service. It’s split into two parts:
- **Images**: Pictures of faces, stored in `s3://your-bucket/dataset/frames/`.
- **Audio**: Voice recordings, stored in `s3://your-bucket/dataset/audio/`.

Each part has three folders:
- `train/`: Data to teach the system.
- `test/`: Data to check how well it learned.
- `validate/`: Extra data to fine-tune it.

Inside those, we have:
- `real/`: Real images or audio.
- `fake/`: Fake (deepfake) images or audio.

The images and audio don’t come from the same videos (they’re separate sources), but that’s okay—we’ll teach the system to spot patterns anyway.

### 2. Getting the Data Ready
We don’t download files to the computer. Instead, we **stream** them from S3 into memory (like watching a video online without saving it). Here’s what we do:
- **Images**: Turn pictures into a standard size (224x224 pixels) and adjust their colors so they’re easier to analyze.
- **Audio**: Turn voice recordings into pictures of sound (called spectrograms, sized 128x128) so the computer can “see” the voice patterns.

### 3. Building the Brain (Models)
The system has three parts, like a detective team:
- **Image Team**: Looks at faces using a tool called ResNet18 (a smart picture analyzer) to guess emotions (happy, sad, angry, etc.).
- **Audio Team**: Listens to voices using a mix of tools (CNN and LSTM) to guess emotions from the sound.
- **Fusion Team**: Combines the clues in three ways (early, mid, late) to decide: Real or Fake?

The system learns by comparing emotions. If the face looks happy but the voice sounds sad, it’s suspicious!

### 4. Mixing the Clues (Fusion)
- **Early Fusion**: Mix the image and audio data right away and let one big brain figure it out.
- **Mid Fusion**: Let the image and audio teams work separately first, then mix their findings in the middle.
- **Late Fusion**: Let each team make a guess, then vote on the final answer.

### 5. Checking Emotions
We add mini-checks to see what emotions the image and audio show. If they don’t match (like a smiling face with a crying voice), the system flags it as a fake.

### 6. Saving Everything
We save **everything** to S3 so you don’t miss a thing:
- **Models**: The trained brains (saved as `.pth` files).
- **Plots**: Pictures showing how well it’s working (saved as `.png`).
- **Logs**: Notes about what happened during training (saved as `.txt` or `.csv`).
- **Intermediate Results**: Halfway findings, like emotion guesses or memory usage.

These go into `s3://your-bucket/outputs/`, which we create automatically if it’s not there.

### 7. Running It
We run this on **AWS EC2**, a cloud computer you control through a terminal (like a text-only interface). You type a command like `python src/train.py --config config.yaml`, and it starts working. No downloading—just streaming and saving to S3.

---

## Folder Structure
Here’s how the project is organized:

### On Your Computer (Code Only)
```
deepfake_detection/
├── src/                    # All the code lives here
│   ├── s3_utils.py        # Tools to talk to S3 (stream data, save files)
│   ├── preprocess.py      # Prepares images and audio in memory
│   ├── models.py          # Defines the brains (image, audio, fusion)
│   ├── train.py           # Runs the training
│   ├── evaluate.py        # Checks how good the system is
│   └── utils.py           # Extra helpers (memory checks, plotting)
├── config/                # Settings
│   ├── config.yaml        # S3 bucket name, settings like batch size
├── requirements.txt       # List of tools we need (like Python libraries)
└── README.md              # This file!
```

### In S3 (Data and Results)
```
s3://your-bucket/
├── dataset/               # Where the data lives
│   ├── frames/           # Images
│   │   ├── train/
│   │   │   ├── real/
│   │   │   └── fake/
│   │   ├── test/
│   │   │   ├── real/
│   │   │   └── fake/
│   │   └── validate/
│   │       ├── real/
│   │       └── fake/
│   ├── audio/            # Audio files
│   │   ├── train/
│   │   │   ├── real/
│   │   │   └── fake/
│   │   ├── test/
│   │   │   ├── real/
│   │   │   └── fake/
│   │   └── validate/
│   │       ├── real/
│   │       └── fake/
├── outputs/              # Where results go (created automatically)
│   ├── models/          # Saved brains (.pth files)
│   ├── plots/           # Pictures of results (.png files)
│   ├── logs/            # Notes about training (.txt/.csv files)
```

---

## How We Keep the Computer Happy (Memory Management)
We don’t want the EC2 computer to crash, so we make sure it never uses more than **80% of its memory** (leaving 20% free). Here’s how:
- **Streaming**: We don’t save files—just process them in memory as they come from S3.
- **Batching**: We work on small groups of data (like 8-16 images/audio at a time) instead of everything at once.
- **Checking Memory**: We use a tool (`psutil`) to watch memory usage. If it gets too high, we stop or shrink the batch.
- **Example**: If EC2 has 4GB of memory:
  - 80% = 3.2GB max.
  - We aim for ~2.9GB (70%) to stay safe.
  - Batch size adjusts automatically to fit.

---

## Pictures We Make (Visualizations)
We’ll create lots of pictures (charts) to show how the system is doing. All have labels, titles, and legends (like a key to explain colors). They’re saved to `s3://your-bucket/outputs/plots/`.

### Before Training
1. **Data Distribution**: How many real vs. fake files we have.
2. **Audio Samples**: Pictures of sound (spectrograms) for real vs. fake.
3. **Image Samples**: A grid of real vs. fake faces.

### During Training
4. **Loss Curves**: How much the system improves over time.
5. **Accuracy Curves**: How often it’s right.
6. **Emotion Consistency Heatmap**: Do emotions match between audio and images?
7. **Confusion Matrix**: What it gets right or wrong (e.g., real called fake).

### After Training
8. **ROC Curve**: How good it is at spotting fakes.
9. **Precision-Recall Curve**: Balancing accuracy and coverage.
10. **F1-Score Plot**: A score mixing precision and recall.
11. **Sensitivity-Specificity Curve**: How it balances true vs. false guesses.
12. **Prediction Distribution**: What the system guesses (real or fake scores).
13. **Precision vs. Recall vs. Threshold**: How changing the cutoff affects results.
14. **Feature Importance**: Which clues (face or voice) matter most.
15. **Memory Usage Plot**: Did we stay under 80%?

### Decision-Making Pictures
16. **Learning Rate Sensitivity**: How fast should it learn?
17. **Batch Size Impact**: Does bigger or smaller groups work better?
18. **Dropout Rate Analysis**: Should we forget some clues to avoid mistakes?
19. **Fusion Comparison**: Which mixing method (early, mid, late) is best?

---

## Tools We Use
- **Python**: The language we write in.
- **PyTorch**: Helps build the brain (models).
- **boto3**: Talks to S3.
- **psutil**: Watches memory.
- **librosa**: Turns audio into pictures.
- **PIL**: Works with images.
- **matplotlib**: Makes our charts.

These are listed in `requirements.txt`—install them on EC2 with `pip install -r requirements.txt`.

---

## How to Run It
1. **Set Up EC2**:
   - Pick an EC2 instance (like `t2.medium` with 4GB RAM).
   - Connect via terminal (SSH).

2. **Upload Code**:
   - Copy the `deepfake_detection/` folder to EC2 (e.g., with `scp`).

3. **Install Tools**:
   - Run `pip install -r requirements.txt` in the terminal.

4. **Edit Config**:
   - Open `config/config.yaml` and add your S3 bucket name (e.g., `s3://my-deepfake-bucket`).

5. **Start Training**:
   - Run `python src/train.py --config config.yaml`.
   - It streams data, trains, and saves everything to S3.

6. **Check Results**:
   - Run `python src/evaluate.py` to see how it did.
   - Look in `s3://your-bucket/outputs/` for models, plots, and logs.

---

## Extra Details
- **Emotions**: We check for emotions like happy, sad, or angry to spot mismatches.
- **Saving Everything**: Every step—models, emotion checks, memory logs—goes to S3.
- **Fusion**: Testing all three ways helps us find the best one.
- **No Crashes**: Memory stays under 80%, so EC2 keeps running smoothly.

---

## What’s Next?
This system is ready to learn from your data and spot deepfakes. You can:
- Add more data to make it smarter.
- Try it on bigger EC2 computers for faster training.
- Use the plots to see what works best.

If anything’s unclear, imagine you’re teaching a friend: every detail here is for you to understand and use this system like a pro!

---

This `README.md` covers the full scope of your project in a way that’s detailed yet beginner-friendly. Let me know if you want to tweak anything or proceed with the code!