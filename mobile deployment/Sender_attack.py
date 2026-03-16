import socket
import pandas as pd
import time
import json
import argparse
import sys


UDP_IP = "192.168.1.0" # <--- REPLACE WITH YOUR PHONE'S IP
UDP_PORT = 5005
ATTACK_START_TIME = 910.0 
SPEED_OFFSET_ATTACK = 18.3 

def run_mixed_scenario(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Filter for the Test Phase (e.g., Data after 800s)
    # We want to show Normal (800-900) THEN Attack (900+)
    test_phase_df = df[df['time'] >= 900.0].sort_values(by='time').copy()
    
    if test_phase_df.empty:
        print("Error: No data found for time >= 900s.")
        return

    print(f"Starting Simulation with {len(test_phase_df)} packets...")
    print(f"Phase 1: Normal Driving (Time < {ATTACK_START_TIME}s)")
    print(f"Phase 2: Speed Injection Attack (Time >= {ATTACK_START_TIME}s)")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    for index, row in test_phase_df.iterrows():
        current_time = row['time']
        
        # ==========================================
        # THE "SWITCH" LOGIC
        # ==========================================
        if current_time < ATTACK_START_TIME:
            # PHASE 1: NORMAL
            # Send the data exactly as it appears in the CSV
            sent_speed = row['speed']
            status = "NORMAL"
        else:
            # PHASE 2: ATTACK
            # Inject the anomaly before sending
            sent_speed = row['speed'] + SPEED_OFFSET_ATTACK
            status = "INJECTING ATTACK"
            
        # Construct the BSM
        bsm = {
            
            "speed": sent_speed,       # Modified only if in attack phase
            "accel": row['acceleration']
        }
        
        # Send Packet
        message = json.dumps(bsm).encode('utf-8')
        sock.sendto(message, (UDP_IP, UDP_PORT))
        
        # Visualization
        sys.stdout.write(f"\rTime: {current_time:.1f}s | {status} | Speed: {sent_speed:.2f} m/s   ")
        sys.stdout.flush()
        
        # Simulate transmission delay (0.05s = 20Hz for smooth demo)
        time.sleep(0.02)

    print("\n\nSimulation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='filtered_data.csv')
    args = parser.parse_args()
    
    run_mixed_scenario(args.data)
