# fertilizer_recommender.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

FERTILIZER_MODEL = None
# Defining the features list for clarity, using the expected CSV names
FERTILIZER_FEATURES = ['N', 'P', 'K', 'Temp', 'Humidity', 'pH', 'soil_encoded'] 
CROP_ENCODER = None 

def load_and_train_fertilizer_model(file_path='Fertilizer_data.csv'):
    """Loads data, cleans it, trains the Fertilizer Recommendation Model, and stores it."""
    global FERTILIZER_MODEL, CROP_ENCODER

    print("\n--- Loading and Training Fertilizer Model ---")
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Fertilizer data file not found at {file_path}. The soil nutrients are crying out for data!")
        return False
    
    # 1. INITIAL DATA CLEANING: Drop any rows that had missing values in the CSV.
    initial_rows = len(data)
    data.dropna(inplace=True) 
    print(f"   Cleaned fertilizer data: Dropped {initial_rows - len(data)} row(s) initially with missing data.")
    
    # 2. FEATURE ENGINEERING: Map soil types to numerical values.
    # Note: Column names MUST be exact for 'Soil Type' and 'Fertilizer Name'
    soil_mapping = {
        'Sandy': 0, 'Loamy': 1, 'Black': 2, 'Red': 3, 'Clayey': 4, 'Alluvial': 5
    }
    
    data['soil_encoded'] = data['Soil Type'].map(soil_mapping)
    
    # 3. SECONDARY DATA CLEANING: Handle unrecognized soil types.
    # If a soil type in the CSV isn't in the map, it results in NaN, which fails the model.
    rows_imputed = data['soil_encoded'].isna().sum()
    data['soil_encoded'].fillna(1, inplace=True) # Fill unrecognized (NaN) with 'Loamy' (1)
    print(f"   Cleaned soil encoding: Imputed {rows_imputed} row(s) with unrecognized soil types to 'Loamy' (1).")

    # 4. TARGET ENCODING
    CROP_ENCODER = LabelEncoder()
    data['fertilizer_encoded'] = CROP_ENCODER.fit_transform(data['Fertilizer Name'])
    
    # 5. SPLIT AND TRAIN
    # X and y selection using the cleaned data
    X = data[['N', 'P', 'K', 'Temp', 'Humidity', 'pH', 'soil_encoded']] 
    y = data['fertilizer_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    
    FERTILIZER_MODEL = KNeighborsClassifier(n_neighbors=5)
    FERTILIZER_MODEL.fit(X_train, y_train) 
    
    # 6. EVALUATION
    accuracy = FERTILIZER_MODEL.score(X_test, y_test)
    print(f"✅ Fertilizer Model trained. Accuracy: {accuracy * 100:.2f}%")
    return True

def recommend_fertilizer(N, P, K, temp, humidity, ph, soil_type):
    """Predicts and returns the best fertilizer based on crop and soil inputs."""
    if FERTILIZER_MODEL is None:
        # This shouldn't happen if load_and_train_fertilizer_model ran successfully
        raise RuntimeError("Fertilizer model not loaded. Call load_and_train_fertilizer_model() first.")

    # Get the mapping again for prediction
    soil_mapping = {
        'Sandy': 0, 'Loamy': 1, 'Black': 2, 'Red': 3, 'Clayey': 4, 'Alluvial': 5
    }
    
    # Use .get() to safely retrieve the encoded value, defaulting to 1 (Loamy) if unknown
    soil_encoded = soil_mapping.get(soil_type, 1)
    
    if soil_encoded == 1 and soil_type not in soil_mapping:
        print(f"Warning: Unknown soil type '{soil_type}'. Defaulting to 'Loamy' (1) for prediction.")

    # Prepare input for the model. Order MUST match FERTILIZER_FEATURES.
    user_input = np.array([[N, P, K, temp, humidity, ph, soil_encoded]])
    
    # Predict and decode
    fertilizer_encoded = FERTILIZER_MODEL.predict(user_input)
    recommended_fertilizer = CROP_ENCODER.inverse_transform(fertilizer_encoded)[0]
    
    return recommended_fertilizer

if __name__ == '__main__':
    # This block is for testing this module independently
    if load_and_train_fertilizer_model():
        # Test with example data (N=20, P=20, K=20, Loamy soil, etc.)
        fert = recommend_fertilizer(N=20, P=20, K=20, temp=25, humidity=60, ph=6.5, soil_type='Loamy')
        print(f"\n--- Standalone Fertilizer Test ---")
        print(f"Recommended Fertilizer: **{fert.upper()}**")