# VAE_Based_Real_Time_Anomaly_Detection_Approach_for_Enhanced_V2X_Communication_Security

[![Paper](https://img.shields.io/badge/Paper-MDPI_Appl._Sci.-blue)](https://doi.org/10.3390/app15126739)
[![Framework](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![Simulation](https://img.shields.io/badge/SUMO-Traffic_Sim-green)](https://eclipse.dev/sumo/)

> **Official PyTorch Implementation** of the paper: *VAE-Based Real-Time Anomaly Detection Approach for Enhanced V2X Communication Security*, published in **Applied Sciences (2025)**.

## 🚀 Project Overview
Vehicle-to-Everything (V2X) communication is critical for Intelligent Transportation Systems (ITS), but it is highly susceptible to cyberattacks that falsify Basic Safety Messages (BSMs). This project implements a **lightweight, real-time anomaly detection framework** capable of identifying data injection attacks (e.g., position/speed falsification) with high precision on resource-constrained devices like On-Board Units (OBUs).

### Key Features
* [cite_start]**Hybrid Architecture:** Combines **Variational Autoencoders (VAE)** for probabilistic reconstruction with **Convolutional Neural Networks (CNN)** for spatial feature extraction[cite: 26].
* [cite_start]**Real-Time Processing:** Uses a sliding window mechanism (window size: 4, stride: 1) to process streaming BSM data[cite: 27, 179].
* [cite_start]**High Performance:** Achieves **0.99 Recall** and **0.95 F1-Score** on complex anomalies like constant position offsets[cite: 28].
* [cite_start]**Ultra-Low Latency:** Average inference time of **1.3 ms (0.0013s)** per window, suitable for safety-critical deployment[cite: 30].

---

## 🏗️ Architecture
The model utilizes a VAE-CNN architecture to capture both temporal and spatial dependencies in vehicle kinematics.

* [cite_start]**Encoder:** Sequence of Convolutional layers + Batch Normalization + Leaky ReLU to compress BSM data into a latent vector[cite: 251, 252].
* [cite_start]**Latent Space:** Probabilistic sampling ($z = \mu + \sigma \cdot \epsilon$) to model normal behavior distributions[cite: 259].
* [cite_start]**Decoder:** Transposed Convolutional layers to reconstruct the input window[cite: 263].

---

## 📊 Performance & Results
[cite_start]The model was evaluated against traditional Autoencoders and VAE-LSTM architectures using a custom dataset generated via **SUMO** (Gangnam, Seoul road network)[cite: 272, 277].

| Anomaly Type | Model | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Constant Position Offset** | **Ours (VAE-CNN)** | **0.91** | **0.99** | **0.95** |
| Constant Speed Offset | Ours (VAE-CNN) | 0.90 | 0.95 | 0.93 |
| Vehicle Speed Offset | Ours (VAE-CNN) | 0.82 | 0.93 | 0.87 |

[cite_start]*Comparison against baseline models (Autoencoder, VAE-LSTM) available in the full paper[cite: 372].*

---

## 🛠️ Installation & Usage

### 1. Prerequisites
* Python 3.8+
* PyTorch 2.x
* SUMO (for data generation)

### 2. Clone the Repository
```bash
git clone [https://github.com/yourusername/V2X-Anomaly-Detection-VAE.git](https://github.com/yourusername/V2X-Anomaly-Detection-VAE.git)
cd V2X-Anomaly-Detection-VAE
pip install -r requirements.txt
