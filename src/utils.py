import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    # (Same as before - strictly data cleaning)
    data = pd.read_csv(file_path)
    dataset = data[data['time'] < 900].reset_index(drop=True)
    train_data = dataset[dataset['time'] < 800].reset_index(drop=True)
    test_data = dataset[dataset['time'] >= 800].reset_index(drop=True)
    
    cols_to_drop = ['angle', 'id', 'time']
    train_data_filtered = train_data.drop(columns=cols_to_drop, errors='ignore')
    test_data_filtered = test_data.drop(columns=cols_to_drop, errors='ignore')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_normalized = pd.DataFrame(scaler.fit_transform(train_data_filtered), columns=train_data_filtered.columns)
    test_normalized = pd.DataFrame(scaler.transform(test_data_filtered), columns=test_data_filtered.columns)
    
    return train_normalized, test_normalized, scaler

def create_sliding_window(data, seq_len=4, model_type='vae'):
    """
    Generates sequences and corrects the shape for PyTorch.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
        
    sequences = []
    # Your exact sequence generation logic
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i + seq_len])
    
    sequences = np.array(sequences) # Shape: (N, Window, Features)
    
    # --- SHAPE CORRECTION LOGIC ---
    if model_type == 'lstm':
        # LSTM wants: (Batch, Window, Features) -> No change needed
        return sequences
    
    elif model_type in ['vae', 'ae']:
        # 1D CNNs want: (Batch, Features, Window) -> Transpose needed
        # We swap the last two dimensions (axis 1 and axis 2)
        return sequences.transpose(0, 2, 1)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_dataloaders(train_df, test_df, batch_size=32, window_size=4, model_type='vae'):
    train_windows = create_sliding_window(train_df, window_size, model_type=model_type)
    test_windows = create_sliding_window(test_df, window_size, model_type=model_type)
    
    train_tensor = torch.tensor(train_windows, dtype=torch.float32)
    test_tensor = torch.tensor(test_windows, dtype=torch.float32)
    
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, test_loader
