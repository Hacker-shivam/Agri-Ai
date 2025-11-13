# crop_recommender.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Global variable to hold the trained model and features
CROP_MODEL = None
CROP_FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall']

def load_and_train_crop_model(file_path='Crop_data.csv'):
    """Loads data, trains the Crop Recommendation Model, and stores it."""
    global CROP_MODEL

    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Crop data file not found at {file_path}. Did you put the 'Crop_data.csv' in the project folder?")
        return False

    X = data[CROP_FEATURES]
    y = data['label'] # 'label' column is the crop name

    # Train a new model (or load from disk in a real application)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    
    CROP_MODEL = RandomForestClassifier(n_estimators=100, random_state=42)
    CROP_MODEL.fit(X_train, y_train)
    
    # Optional: Print accuracy on test set
    accuracy = CROP_MODEL.score(X_test, y_test)
    print(f"✅ Crop Model trained. Accuracy: {accuracy * 100:.2f}%")
    return True

def recommend_crop(N, P, K, temp, hum, ph, rain):
    """Predicts and returns the best crop based on input parameters."""
    if CROP_MODEL is None:
        raise RuntimeError("Crop model not loaded. Call load_and_train_crop_model() first.")
    
    # Ensure input is in the correct format for the model
    user_input = np.array([[N, P, K, temp, hum, ph, rain]])
    
    recommended_crop = CROP_MODEL.predict(user_input)[0]
    
    # Get confidence (optional)
    probabilities = CROP_MODEL.predict_proba(user_input)
    confidence = probabilities.max() * 100
    
    return recommended_crop, confidence

if __name__ == '__main__':
    # This block is for testing this module independently
    if load_and_train_crop_model():
        # Test with example data (adjust these values to see different results!)
        crop, conf = recommend_crop(N=90, P=40, K=40, temp=20.0, hum=80.0, ph=6.0, rain=200.0)
        print(f"\n--- Standalone Crop Test ---")
        print(f"Recommended Crop: **{crop.upper()}** (Confidence: {conf:.2f}%)")