# VAE-Based Real-Time Anomaly Detection for V2X Security

[![Paper](https://img.shields.io/badge/Paper-MDPI_Appl._Sci.-blue)](https://doi.org/10.3390/app15126739)
[![Framework](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Official PyTorch Implementation** of the paper: *VAE-Based Real-Time Anomaly Detection Approach for Enhanced V2X Communication Security*, published in **Applied Sciences (2025)**.

## 🚀 Project Overview
Vehicle-to-Everything (V2X) communication is critical for Intelligent Transportation Systems (ITS), but it is vulnerable to cyberattacks that falsify Basic Safety Messages (BSMs). This repository contains a **lightweight, real-time anomaly detection framework** capable of identifying data injection attacks (e.g., position/speed falsification) on resource-constrained devices like On-Board Units (OBUs).

```figure
![Figure 1](https://github.com/user-attachments/assets/f339b9d8-5a63-4473-8ad1-730ef4cd6f62)
```

### Key Features
* **Hybrid Architecture:** Uses a **1D Convolutional VAE** to capture temporal dependencies with low computational cost.
* **Smart Reshaping:** Custom data pipeline that automatically adjusts input shapes for CNNs `(Batch, Features, Window)` vs. LSTMs `(Batch, Window, Features)`.
* **Reproducible Baselines:** Includes full implementations of **Autoencoder** and **LSTM** models for benchmarking.
* **High Performance:** Achieves **0.99 Recall** and **1.3ms inference time**, outperforming traditional methods.

---

## 📂 Repository Structure

```text
V2X-Anomaly-Detection-VAE/
│
├── simulation/              
│   ├── additional.add.xml
│   ├── demand.rou.xml
│   ├── osm.net.xml.gz           
|   ├── osm.netccfg
|   ├── osm.passenger.trips.xml
|   ├── osm.sumocfg
│   ├── osm.view.xml        
│   └── parse_fcd.py         
│
├── src/
│   ├── model.py         # The proposed VAE-CNN Architecture
│   ├── baselines.py     # Comparison models (Autoencoder, LSTM)
│   ├── train.py         
│   ├── evaluate.py      
│   └── utils.py         
│
├── assets/              
├── requirements.txt     
└── README.md            

```
---
## 🚗 Data Generation (SUMO Simulation)
To ensure reproducibility, this repository includes the full simulation setup used to generate the dataset.

**Scenario Details:**
* **Location:** Gangnam District, Seoul (Imported from OpenStreetMap)
* **Traffic Density:** ~680 vehicles over 1000s simulation time
* **Collection Logic:** Data is logged only from vehicles within a **500m radius** of the observer vehicle ($V_0$).

**How to Run:**
1.  Install [SUMO](https://eclipse.dev/sumo/).
2.  Navigate to the simulation folder:
    ```bash
    cd simulation
    ```
3.  Run the simulation to generate raw FCD (Floating Car Data):
    ```bash
    sumo -c config.sumocfg --fcd-output raw_trace.xml
    ```
4. Process the Data
We use a custom script to parse the XML and apply the **500m communication radius** constraint.

```bash
# Convert raw XML to the final training CSV
python simulation/parse_fcd.py --input simulation/fcd_out.xml --output data/dataset.csv --ego veh0 --radius 500
```
---

## 🏗️ Model Architecture

The model processes streaming BSM data using a sliding window mechanism. It utilizes 1D Convolutions to extract features from time-series data (Speed, Acceleration, Position) and a Variational Autoencoder (VAE) to learn the probabilistic distribution of normal driving behavior.

<img width="1028" height="503" alt="figure 2" src="https://github.com/user-attachments/assets/d320077d-3be5-44fc-88e4-60a3c4272789" />

**Benchmarking**
We compare our approach against two industry standards:
1. **Standard Autoencoder (AE):** A deterministic reconstruction model using 1D CNNs.
2. **VAE-LSTM:** A recurrent generative model often used for time-series anomaly detection.

See src/baselines.py for implementation details.

---

## 📊 Performance & Results
The proposed VAE-CNN outperforms baseline models, particularly in detecting complex Vehicle-Targeted Speed Offsets, while maintaining the lowest inference time.

| Anomaly Type | Model | Precision | Recall | Accuracy | F1-Score | S.F | Inference Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Constant Position Offset** | **Ours (VAE-CNN)** | **0.91** | **0.99** | **0.97** | **0.95** | 0.5 | **0.0013s** |
||VAE-LSTM | 0.89 |	0.99 |	0.96 | 0.94 | 0.5 | 0.0029s |
||Autoencoder |	0.84 | 0.99 | 0.94 | 0.91 | 0.6 | 0.0010s |
| **Constant Speed Offset** | **Ours (VAE-CNN)** | **0.90** | **0.95** | **0.95** | **0.93** | **0.6** | **0.0014s** |
||VAE-LSTM | 0.88 |	0.93 |	0.94 | 0.91 | 0.6 | 0.0034s |
||Autoencoder |	0.80 | 0.73 | 0.86 | 0.76 | 0.6 | 0.0014s |


Comparison against baseline models (Autoencoder, VAE-LSTM), details available in the full paper.

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
```
### 3. 2. Train the Model
You can train the proposed model or the baselines using the --model flag:
```bash
# Train the Proposed VAE-CNN
python src/train.py --data dataset.csv --model vae

# Train the Baseline Autoencoder
python src/train.py --data dataset.csv --model ae

# Train the Baseline LSTM
python src/train.py --data dataset.csv --model lstm
```
### 4. Evaluate & Visualize
Run the evaluation script to calculate F1-scores and generate reconstruction error histograms (saved to assets/).
```bash
python src/evaluate.py --data dataset.csv --model vae_v2x_model.pth
```
---

## 🔗 Citation
If you use this code or dataset in your research, please cite our paper:
```text
@article{gebrezgiher2025vae,
  title={VAE-Based Real-Time Anomaly Detection Approach for Enhanced V2X Communication Security},
  author={Gebrezgiher, Yonas Teweldemedhin and Jeremiah, Sekione Reward and Gritzalis, Stefanos and Park, Jong Hyuk},
  journal={Applied Sciences},
  volume={15},
  number={12},
  pages={6739},
  year={2025},
  publisher={MDPI},
  doi={10.3390/app15126739}
}
```
Research conducted at the Department of Computer Science and Engineering, Seoul National University of Science and Technology.
