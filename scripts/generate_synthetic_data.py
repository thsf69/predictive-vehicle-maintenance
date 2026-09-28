import pandas as pd
import numpy as np
import os
import argparse

def generate_synthetic_vehicle_data(num_records=10000, random_seed=42):
    """
    Generates synthetic vehicle telemetry data for predictive maintenance.
    """
    np.random.seed(random_seed)
    
    # 1. Base numerical features
    operating_hours = np.random.uniform(100, 15000, num_records)
    vehicle_speed = np.random.uniform(0, 130, num_records)
    
    # Baseline for other features
    engine_temperature = np.random.normal(90, 5, num_records)
    rpm = np.random.normal(2000, 500, num_records)
    oil_pressure = np.random.normal(40, 5, num_records)
    vibration_level = np.random.normal(0.3, 0.1, num_records)
    battery_voltage = np.random.normal(13.5, 0.5, num_records)
    
    # Introduce conditions that lead to maintenance risk
    # Probability of risk increases with operating hours
    base_risk = (operating_hours / 15000) * 0.2
    
    # Let's create an anomaly factor for a subset of records (~20%)
    is_anomalous = np.random.rand(num_records) < 0.2
    
    # Modify features for anomalous records to simulate wear/tear
    engine_temperature[is_anomalous] += np.random.normal(15, 5, sum(is_anomalous))
    rpm[is_anomalous] += np.random.normal(1000, 300, sum(is_anomalous))
    oil_pressure[is_anomalous] -= np.random.normal(15, 5, sum(is_anomalous))
    vibration_level[is_anomalous] += np.random.normal(0.5, 0.2, sum(is_anomalous))
    battery_voltage[is_anomalous] -= np.random.normal(2, 0.5, sum(is_anomalous))
    
    # Calculate risk score based on thresholds and interactions
    risk_score = base_risk.copy()
    risk_score += np.where(engine_temperature > 105, 0.3, 0)
    risk_score += np.where(rpm > 3500, 0.1, 0)
    risk_score += np.where(oil_pressure < 25, 0.3, 0)
    risk_score += np.where(vibration_level > 0.7, 0.2, 0)
    risk_score += np.where(battery_voltage < 12.0, 0.1, 0)
    
    # Add some random noise to risk score
    risk_score += np.random.normal(0, 0.1, num_records)
    
    # Binarize target based on a threshold to get roughly 15-20% positive class
    # High risk = 1, Low risk = 0
    target_threshold = 0.6
    maintenance_risk = (risk_score >= target_threshold).astype(int)
    
    # Add categorical feature for vehicle type
    vehicle_types = ['Sedan', 'SUV', 'Truck', 'Van']
    vehicle_type = np.random.choice(vehicle_types, num_records)
    
    # Introduce some missing values to test preprocessing pipeline (approx 1%)
    def add_missing(arr, prob=0.01):
        mask = np.random.rand(len(arr)) < prob
        arr = arr.astype(float)
        arr[mask] = np.nan
        return arr
        
    engine_temperature = add_missing(engine_temperature)
    oil_pressure = add_missing(oil_pressure)
    
    # Assemble dataframe
    df = pd.DataFrame({
        'vehicle_id': [f'V_{i:05d}' for i in range(num_records)],
        'vehicle_type': vehicle_type,
        'operating_hours': operating_hours,
        'engine_temperature': engine_temperature,
        'rpm': rpm,
        'vehicle_speed': vehicle_speed,
        'oil_pressure': oil_pressure,
        'vibration_level': vibration_level,
        'battery_voltage': battery_voltage,
        'maintenance_risk': maintenance_risk
    })
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic vehicle data.")
    parser.add_argument('--output', type=str, default='data/raw/synthetic_vehicle_data.csv', help='Output path')
    parser.add_argument('--records', type=int, default=10000, help='Number of records')
    args = parser.parse_args()
    
    print(f"Generating {args.records} synthetic records...")
    df = generate_synthetic_vehicle_data(num_records=args.records)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    df.to_csv(args.output, index=False)
    print(f"Data saved to {args.output}")
    print(f"Class balance:\n{df['maintenance_risk'].value_counts(normalize=True)}")

if __name__ == "__main__":
    main()
