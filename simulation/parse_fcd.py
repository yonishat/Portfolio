import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import argparse
import os

def parse_fcd_xml(xml_file):
    """
    Parses SUMO FCD (Floating Car Data) XML output into a Pandas DataFrame.
    """
    print(f"Parsing XML file: {xml_file}...")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    # Parse XML to extract vehicle data
    for timestep in root.findall('timestep'):
        time = float(timestep.get('time'))
        for vehicle in timestep.findall('vehicle'):
            data.append({
                'time': time,
                'id': vehicle.get('id'),
                'x': float(vehicle.get('x')),
                'y': float(vehicle.get('y')),
                'speed': float(vehicle.get('speed')),
                'angle': float(vehicle.get('angle')),
                'acceleration': float(vehicle.get('acceleration') or 0.0), # Handle missing accel if any
            })
    
    return pd.DataFrame(data)

def filter_by_radius(df, ego_id='veh0', radius=500):
    """
    Filters vehicles that are within 'radius' meters of the 'ego_id' vehicle.
    Optimized with vectorization for performance.
    """
    print(f"Filtering data: Radius {radius}m around '{ego_id}'...")
    
    # 1. Extract the Ego Vehicle's trajectory
    ego_data = df[df['id'] == ego_id][['time', 'x', 'y']].rename(
        columns={'x': 'ego_x', 'y': 'ego_y'}
    )
    
    if ego_data.empty:
        raise ValueError(f"Ego vehicle '{ego_id}' not found in the dataset!")

    # 2. Merge Ego position onto the main dataframe based on 'time'
    # This aligns every vehicle's position with the Ego's position at that exact timestamp
    merged = pd.merge(df, ego_data, on='time', how='inner')

    # 3. Vectorized Distance Calculation (Euclidean)
    # Much faster than iterating row-by-row
    merged['distance'] = np.sqrt(
        (merged['x'] - merged['ego_x'])**2 + 
        (merged['y'] - merged['ego_y'])**2
    )

    # 4. Apply Filter
    # Keep vehicles within radius, excluding the ego vehicle itself
    filtered_df = merged[
        (merged['distance'] <= radius) & 
        (merged['id'] != ego_id)
    ].copy()

    # Drop helper columns to clean up
    filtered_df = filtered_df.drop(columns=['ego_x', 'ego_y', 'distance'])
    
    print(f"Filter Complete. Reduced {len(df)} rows to {len(filtered_df)} rows.")
    return filtered_df

def main():
    parser = argparse.ArgumentParser(description="Process SUMO FCD XML into a filtered CSV dataset.")
    parser.add_argument('--input', type=str, required=True, help='Path to input .xml file (e.g., fcd_out.xml)')
    parser.add_argument('--output', type=str, default='dataset.csv', help='Path to output .csv file')
    parser.add_argument('--ego', type=str, default='veh0', help='ID of the data collector vehicle (default: veh0)')
    parser.add_argument('--radius', type=float, default=500, help='Communication radius in meters (default: 500)')
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # Run Pipeline
    df = parse_fcd_xml(args.input)
    filtered_df = filter_by_radius(df, ego_id=args.ego, radius=args.radius)
    
    # Save
    filtered_df.to_csv(args.output, index=False)
    print(f"Successfully saved processed dataset to: {args.output}")

if __name__ == "__main__":
    main()
