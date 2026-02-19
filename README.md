📌 Adaptive RGB-Based Face Anti-Spoofing Framework
🔐 Adaptive Multi-Expert Face Anti-Spoofing Using Reliability-Aware Gated Fusion and Domain Adaptation
📖 Overview

This project presents an Adaptive Multi-Expert RGB-Based Face Anti-Spoofing Framework designed to detect face presentation attacks (spoofing) using only a standard RGB camera.

Unlike conventional single-stream CNN models, this framework decomposes spoof detection into three complementary perceptual domains:

RGB Appearance Analysis

Depth-Aware Structural Estimation

Frequency-Domain Artifact Detection

These expert branches are dynamically integrated using a Reliability-Aware Gated Fusion mechanism, followed by an Adaptive Replay Buffer for continual learning under domain shift.

The system achieves:

99.99% Accuracy (CASIA-FASD)

ACER = 0.0164% (Intra-dataset)

92.09% Cross-Dataset Accuracy (CASIA → NUAA)

8× reduction in cross-domain error after adaptation

🚨 Problem Statement

Face recognition systems are vulnerable to:

Printed photo attacks

Replay attacks (video displayed on screen)

Digital spoof artifacts

Traditional methods:

Overfit to specific datasets

Fail under domain shift

Require expensive depth/IR sensors

This project proposes a hardware-independent RGB-only solution with strong generalization and adaptive learning capability.

🧠 System Architecture

The framework consists of four major modules:

1️⃣ Multi-Expert Feature Decomposition
A. RGB Appearance Expert

Backbone: MobileNetV2

Extracts spatial texture cues

Detects color distortions and surface artifacts

B. Depth-Aware Structural Expert

U-Net style encoder-decoder

Predicts pseudo-depth map from RGB

Learns 3D facial structure vs planar spoof surfaces

C. Frequency-Domain Expert

Applies Fast Fourier Transform (FFT)

Extracts spectral artifacts (Moiré patterns, display noise)

2️⃣ Reliability-Aware Gated Fusion

Instead of static concatenation, the system:

Learns confidence weights for each expert

Applies Softmax normalization

Computes dynamically weighted fused embedding:

F_fused = α_rgb * F_rgb + α_depth * F_depth + α_freq * F_freq


This improves robustness under:

Low lighting

Motion blur

Sensor variations

3️⃣ Adaptive Replay Buffer

To handle domain shift:

Stores high-confidence predictions (p > 0.95)

Periodically fine-tunes fusion layers

Reduces catastrophic forgetting

Enables continual learning

📊 Experimental Results
📌 Intra-Dataset (CASIA-FASD)
Metric	Value
Accuracy	99.99%
AUC	100%
EER	0.0000%
APCER	0.0000%
BPCER	0.0327%
ACER	0.0164%
📌 Cross-Dataset (CASIA → NUAA)
Metric	Value
Accuracy	92.09%
AUC	98.14%
ACER	7.56%

Domain adaptation reduced ACER from ~62% to 7.56%.

🛠️ Technology Stack
Programming Language

Python 3.10

Deep Learning Framework

PyTorch

Architectures

MobileNetV2 (Appearance Expert)

U-Net (Depth Expert)

Custom CNN (Frequency Expert)

MLP (Confidence Gating Network)

Optimization

AdamW Optimizer

Learning Rate: 1e-4

Batch Size: 32

Replay Buffer Size: 2000

Hardware

CUDA-enabled GPU recommended

📂 Project Structure
adaptive-face-anti-spoofing/
│
├── datasets/
├── models/
│   ├── rgb_expert.py
│   ├── depth_expert.py
│   ├── frequency_expert.py
│   ├── gated_fusion.py
│
├── training/
│   ├── train.py
│   ├── loss_functions.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│
├── replay_buffer/
│   ├── adaptive_buffer.py
│
├── utils/
│   ├── fft_utils.py
│   ├── preprocessing.py
│
├── requirements.txt
└── README.md

⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/your-username/adaptive-face-anti-spoofing.git
cd adaptive-face-anti-spoofing

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

📦 Dataset Preparation

Due to licensing restrictions, datasets must be downloaded separately:

CASIA-FASD

NUAA

Place dataset folders inside:

datasets/


Follow dataset structure as:

datasets/CASIA/
datasets/NUAA/

🚀 Training
python training/train.py --dataset CASIA --epochs 5

🧪 Evaluation
Intra-dataset testing:
python evaluation/evaluate.py --dataset CASIA

Cross-dataset testing:
python evaluation/evaluate.py --train CASIA --test NUAA

📈 Metrics Reported

Accuracy

Precision

Recall

F1-Score

AUC

EER

APCER

BPCER

ACER

Evaluation follows ISO/IEC 30107-3 biometric standard.

🔄 Adaptive Replay Learning

To enable domain adaptation:

python training/adaptive_update.py --buffer_size 2000 --threshold 0.95

🔍 Visualization

The project includes:

Confusion Matrix

ROC Curve

t-SNE Feature Space Visualization

Pseudo-depth Map Visualization

FFT Magnitude Spectrum Visualization

🔐 Security & Ethical Considerations

Designed for biometric authentication systems

Supports privacy-aware deployment

Does not store facial images permanently

Compatible with GDPR-compliant systems

🚀 Deployment Scope

Suitable for:

Mobile authentication

Banking security

Smart access control

Identity verification systems

📌 Limitations

Sensitive to extreme low-light conditions

Limited robustness against high-fidelity 3D mask attacks

Requires GPU for training

Future work includes model compression and multi-sensor integration.

📚 Citation

If you use this work, please cite:

Vikas Kumar,
Adaptive RGB-Based Face Spoof Detection Using Multi-Expert Feature Decomposition and Gated Fusion.

👨‍💻 Author

Vikas Kumar
Department of Computer Engineering
Army Institute of Technology, Pune
Email: Vikaskumar_240252@aitpune.edu.in

⭐ Final Note

This project demonstrates that adaptive multi-expert learning combined with domain-aware training provides a scalable and hardware-independent solution for next-generation face anti-spoofing systems.
