import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import Conv_VAE
from baselines import Autoencoder, VAE_LSTM
from utils import load_and_preprocess_data, get_dataloaders
import argparse

# Loss Function
def loss_function(recon, x, mu, logvar, beta=1.0):
    RECON = F.mse_loss(recon, x, reduction='sum') / x.size(0)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return RECON + beta * KLD

def train_model(data_path, epochs=100, batch_size=32, lr=0.001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    # 1. Prepare Data
    train_df, test_df, _ = load_and_preprocess_data(data_path)
    train_loader, test_loader = get_dataloaders(train_df, test_df, batch_size=batch_size)
    
    # --- SELECT MODEL BASED ON ARGUMENT ---
    if model_type == 'vae':
        model = Conv_VAE().to(device)
        print("Training Model: VAE-CNN (Proposed)")
    elif model_type == 'ae':
        model = Autoencoder().to(device)
        print("Training Model: Standard Autoencoder (Baseline)")
    elif model_type == 'lstm':
        model = VAE_LSTM().to(device)
        print("Training Model: VAE-LSTM (Baseline)")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    # 3. Training Loop
    best_loss = float('inf')
    patience = 20
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss = loss_function(recon, batch, mu, logvar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss = loss_function(recon, batch, mu, logvar)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early Stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "vae_v2x_model.pth") # Save best model
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print("Training Complete. Model saved as 'vae_v2x_model.pth'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to dataset CSV')
    # Add this new argument:
    parser.add_argument('--model', type=str, default='vae', choices=['vae', 'ae', 'lstm'], help='Choose model: vae, ae, or lstm')
    
    args = parser.parse_args()
    
    # Pass the model type to the train function
    train_model(args.data, model_type=args.model)
