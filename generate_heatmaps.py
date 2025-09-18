import joblib
import numpy as np
import pandas as pd
import json
from itertools import combinations

def generate_heatmaps(model_path='random_forest_sum_severity032.pkl', output_file='all_heatmaps.json'):
    """
    Generates a JSON file containing heatmaps for all pairs of astronomical features.

    Args:
        model_path (str): The file path to the trained machine learning model.
        output_file (str): The file path to save the generated JSON.
    """
    try:
        # Load the trained model
        model = joblib.load(model_path)
        print(f"Model loaded successfully from '{model_path}'.")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Correct list of all astronomical features used by the model
    # The order has been corrected to match the trained model's feature order
    features = [
        'sun_longitude', 'moon_longitude', 'mercury_longitude', 'venus_longitude', 
        'mars_longitude', 'jupiter_longitude', 'saturn_longitude', 'uranus_longitude', 
        'neptune_longitude', 'pluto_longitude', 'north_node_longitude',
        'south_node_longitude', 'ascendant', 'moon_phase', 'mercury_retrograde',
        'eclipse_influence', 'eclipse_strength'
    ]

    # Calculate a neutral 'baseline' for all features to use for constant values
    # For a real application, these would be the mean/median of your training data.
    baseline_values = {
        'sun_longitude': 180, 'moon_longitude': 180, 'mercury_longitude': 180, 'moon_phase': 0.5,
        'venus_longitude': 180, 'mars_longitude': 180, 'jupiter_longitude': 180, 'saturn_longitude': 180,
        'uranus_longitude': 180, 'neptune_longitude': 180, 'pluto_longitude': 180, 'north_node_longitude': 180,
        'south_node_longitude': 180, 'ascendant': 180, 'mercury_retrograde': 0,
        'eclipse_influence': 0, 'eclipse_strength': 0
    }

    # Use a high-resolution grid for smooth heatmaps
    grid_size = 20
    
    all_heatmaps = {}
    
    # Generate all unique pairs of features
    feature_pairs = list(combinations(features, 2))
    total_pairs = len(feature_pairs)
    
    print(f"Generating heatmaps for {total_pairs} unique feature pairs...")

    for i, (feature1, feature2) in enumerate(feature_pairs):
        # Create grid points for the two features
        x = np.linspace(0, 360, grid_size)
        y = np.linspace(0, 360, grid_size)
        
        # Adjust range for specific features
        if 'moon_phase' in [feature1, feature2]:
            x = np.linspace(0, 1, grid_size)
        if 'mercury_retrograde' in [feature1, feature2]:
            x = np.linspace(0, 1, grid_size)
        if 'eclipse_influence' in [feature1, feature2]:
            x = np.linspace(0, 300, grid_size)
        if 'eclipse_strength' in [feature1, feature2]:
            x = np.linspace(0, 1, grid_size)

        # Populate the DataFrame with the grid points for the current pair
        rows = []
        for val_y in y:
            for val_x in x:
                row = baseline_values.copy()
                row[feature1] = val_x
                row[feature2] = val_y
                rows.append(row)
        
        # Create the DataFrame with the correct column order to prevent the ValueError
        df_predict = pd.DataFrame(rows, columns=features)
        
        # Make predictions
        predictions = model.predict(df_predict)
        
        # Reshape predictions into a 2D array for the heatmap
        heatmap_data = predictions.reshape(grid_size, grid_size).tolist()

        # Store the heatmap data in the dictionary
        key = f"{feature1}_vs_{feature2}"
        all_heatmaps[key] = heatmap_data
        
        print(f"[{i+1}/{total_pairs}] Generated heatmap for '{key}'")

    # Save the dictionary to a JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(all_heatmaps, f)
        print(f"\nSuccessfully saved all heatmaps to '{output_file}'.")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == '__main__':
    generate_heatmaps()
