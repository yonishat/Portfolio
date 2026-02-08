import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from model import Conv_VAE
from utils import load_and_preprocess_data, create_sliding_window
import argparse
import os

def get_reconstruction_errors(model, data_loader, device):
    """Passes data through model and returns reconstruction errors."""
    model.eval()
    errors = []
    with torch.no_grad():
        for window in data_loader:
            window = window.to(device)
            recon, mu, logvar = model(window)
            
            # Calculate MSE per window
            loss = F.mse_loss(recon, window, reduction='none')
            # Sum over features and time steps to get scalar error per window
            loss = loss.view(loss.size(0), -1).mean(dim=1)
            errors.extend(loss.cpu().numpy())
    return np.array(errors)

def inject_anomaly(df, anomaly_type='speed', factor=0.5, target_ratio=0.3):
    """
    Injects anomalies into the dataset similar to the paper's methodology.
    """
    modified_df = df.copy()
    num_rows = len(df)
    start_idx = int(num_rows * (1 - target_ratio) / 2)
    end_idx = int(num_rows * (1 + target_ratio) / 2)
    
    # Identify column index
    if anomaly_type == 'speed':
        col_name = 'speed' # Adjust based on your actual CSV column name
    elif anomaly_type == 'position':
        col_name = 'x' # Applying to X position as example
    else:
        raise ValueError("Unknown anomaly type")
    
    # Check if column exists, else use index 0 (fallback)
    col_idx = df.columns.get_loc(col_name) if col_name in df.columns else 0
    max_val = df.iloc[:, col_idx].max()
    
    # Inject Constant Offset
    print(f"Injecting {anomaly_type} anomaly from index {start_idx} to {end_idx}...")
    modified_df.iloc[start_idx:end_idx, col_idx] += (max_val * factor)
    
    # Create Ground Truth labels (0 = Normal, 1 = Anomaly)
    labels = np.zeros(len(modified_df) - 3) # -3 due to window size 4
    # Mark the windows that contain anomalous rows as 1
    # (Simplified logic: if window starts in anomaly region)
    labels[start_idx:end_idx] = 1 
    
    return modified_df, labels

def evaluate_model(data_path, model_path, output_image="assets/anomaly_plot.png"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Data & Model
    train_df, test_df, _ = load_and_preprocess_data(data_path)
    
    model = Conv_VAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")

    # 2. Determine Threshold (98th percentile of Training Data)
    train_windows = create_sliding_window(train_df)
    train_tensor = torch.tensor(train_windows, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=32, shuffle=False)
    
    train_errors = get_reconstruction_errors(model, train_loader, device)
    threshold = np.percentile(train_errors, 98)
    print(f"Anomaly Threshold set at: {threshold:.4f}")

    # 3. Inject Anomalies into Test Data
    anomalous_df, true_labels = inject_anomaly(test_df, anomaly_type='speed', factor=0.6)
    
    # 4. Get Errors for Anomalous Data
    test_windows = create_sliding_window(anomalous_df)
    test_tensor = torch.tensor(test_windows, dtype=torch.float32)
    test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=32, shuffle=False)
    
    test_errors = get_reconstruction_errors(model, test_loader, device)

    # 5. Calculate Metrics
    # Ensure labels match test_errors length
    true_labels = true_labels[:len(test_errors)] 
    
    pred_labels = (test_errors > threshold).astype(int)
    
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    print("\n--- Evaluation Results ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # 6. Plotting (Reproducing Figure 3/4 from Paper)
    plt.figure(figsize=(10, 5))
    
    # Plot Normal Errors (Blue)
    normal_errors = test_errors[true_labels == 0]
    plt.hist(normal_errors, bins=50, alpha=0.6, label='Normal', color='blue')
    
    # Plot Anomalous Errors (Red)
    anom_errors = test_errors[true_labels == 1]
    plt.hist(anom_errors, bins=50, alpha=0.6, label='Anomalous', color='red')
    
    plt.axvline(threshold, color='k', linestyle='dashed', linewidth=1, label='Threshold')
    plt.title('Reconstruction Error Distribution (Speed Anomaly)')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Ensure assets directory exists
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image)
    print(f"\nPlot saved to {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--model', type=str, default='vae_v2x_model.pth', help='Path to trained model')
    args = parser.parse_args()
    
    evaluate_model(args.data, args.model)
