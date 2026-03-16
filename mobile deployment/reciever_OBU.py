import socket
import json
import torch
import numpy as np
import time
from model import Conv_VAE  # Ensure model.py is in the same folder on your phone


# You must fill these with the REAL max values from your training data!
# Check your train.py or dataset.csv to get these.
MAX_SPEED = 36.57
MAX_ACCEL = 2.6


# Your threshold from 'evaluate.py'
ANOMALY_THRESHOLD = 0.05

# Network Config (Listen to All)
UDP_IP = "0.0.0.0" 
UDP_PORT = 5005


print("Loading Lightweight VAE Model...")
device = 'cpu' # because Phones use CPU
model = Conv_VAE()
# You need to copy 'vae_v2x_model.pth' to your phone too!
try:
    model.load_state_dict(torch.		   load("vae_v2x_model.pth", map_location=device))
    model.eval()
    print("Model Loaded Successfully!")
except:
    print("Warning: Model file not found. Running with random weights for demo.")


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

buffer = [] # This is your Sliding Window

print(f"OBU Active. Listening on Port {UDP_PORT}...")

while True:
    # A. RECEIVE RAW PACKET
    data, addr = sock.recvfrom(1024)
    raw_msg = json.loads(data.decode())
    
    # B. REAL-TIME PREPROCESSING (On Device)
    # Normalize the raw values to [0, 1] range
    norm_speed = raw_msg['speed'] / MAX_SPEED
    norm_accel = raw_msg['accel'] / MAX_ACCEL
    #norm_x = raw_msg['x'] / MAX_X
    #norm_y = raw_msg['y'] / MAX_Y
    
    # Create feature vector [x, y, speed, accel]
    # (Order must match your training!)
    features = [norm_speed, norm_accel]
    
    # C. SLIDING WINDOW LOGIC
    buffer.append(features)
    if len(buffer) > 4: # Window size 4
        buffer.pop(0)
        
    # D. INFERENCE TRIGGER
    if len(buffer) == 4:
        start_time = time.time()
        
        # 1. Convert window to Tensor
        # Shape: (1, 4, 4) -> (Batch, Features, Window) because it's CNN
        input_array = np.array(buffer).T # Transpose to get (Features, Window)
        input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)
        
        # 2. Run Model
        with torch.no_grad():
            recon, _, _ = model(input_tensor)
            
        # 3. Calculate Error (MSE)
        loss = torch.mean((recon - input_tensor) ** 2).item()
        
        inference_time = (time.time() - start_time) * 1000 # ms
        
        # E. ALERT LOGIC
        status = "✅ NORMAL"
        if loss > ANOMALY_THRESHOLD:
            status = "🚨 ATTACK DETECTED"
            
        print(f"{status} | Loss: {loss:.5f} | Time: {inference_time:.2f}ms")
