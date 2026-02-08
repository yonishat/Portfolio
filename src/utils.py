import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    """Loads CSV, splits into train/test, and normalizes."""
    data = pd.read_csv(file_path)
    
    # Split based on time 
    dataset = data[data['time'] < 900].reset_index(drop=True)
    train_data = dataset[dataset['time'] < 800].reset_index(drop=True)
    test_data = dataset[dataset['time'] >= 800].reset_index(drop=True)
    
    # Drop unused columns
    cols_to_drop = ['angle', 'id', 'time']
    train_data_filtered = train_data.drop(columns=cols_to_drop, errors='ignore')
    test_data_filtered = test_data.drop(columns=cols_to_drop, errors='ignore')
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_normalized = pd.DataFrame(scaler.fit_transform(train_data_filtered), columns=train_data_filtered.columns)
    test_normalized = pd.DataFrame(scaler.transform(test_data_filtered), columns=test_data_filtered.columns)
    
    return train_normalized, test_normalized, scaler

def create_sliding_window(data, window_size=4, stride=1):
    """Converts dataframe to sliding window numpy array."""
    if isinstance(data, pd.DataFrame):
        data = data.values
        
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i + window_size]
        windows.append(window)
        
    # Reshape for CNN: (Samples, Channels, Height, Width) -> (N, 1, Window, Features)
    windows = np.array(windows)
    return windows.reshape(windows.shape[0], 1, windows.shape[1], windows.shape[2])

def get_dataloaders(train_df, test_df, batch_size=32, window_size=4):
    """Creates PyTorch DataLoaders."""
    train_windows = create_sliding_window(train_df, window_size)
    test_windows = create_sliding_window(test_df, window_size)
    
    train_tensor = torch.tensor(train_windows, dtype=torch.float32)
    test_tensor = torch.tensor(test_windows, dtype=torch.float32)
    
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, test_loader
