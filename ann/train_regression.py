import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Configuration
DATA_DIR = 'data'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

def load_and_clean_data():
    print("Loading datasets...", flush=True)
    # Load attendances
    df_att = pd.read_csv(os.path.join(DATA_DIR, 'attendances.csv'))
    print(f"Loaded attendances: {len(df_att)} rows", flush=True)
    
    # Sample for speed if large
    if len(df_att) > 100000:
        print("Sampling 100,000 rows for faster training...", flush=True)
        df_att = df_att.sample(100000, random_state=42)
    
    # Load emergency rooms
    df_er = pd.read_csv(os.path.join(DATA_DIR, 'emergencyRooms.csv'))
    print("Loaded ER data", flush=True)
    
    # Load COVID risk data
    df_covid = pd.read_csv(os.path.join(DATA_DIR, 'COVIDLocalRiskValuation.csv'))
    print("Loaded COVID data", flush=True)
    
    # Merge attendance with ER metadata
    df = pd.merge(df_att, df_er[['emergency_room', 'is_pediatric']], on='emergency_room', how='left')
    print("Merged ER metadata", flush=True)
    
    # Convert timestamps
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df_covid['dt'] = pd.to_datetime(df_covid['timestamp'], unit='ms').dt.normalize()
    print("Converted timestamps", flush=True)
    
    # Merge with COVID data
    df['date_only'] = df['dt'].dt.normalize()
    df = pd.merge(df, df_covid[['dt', 'risk']], left_on='date_only', right_on='dt', how='left')
    df['risk'] = df['risk'].fillna('low risk')
    print("Merged COVID data", flush=True)
    
    # Feature engineering
    df['hour'] = df['dt_x'].dt.hour
    df['day'] = df['dt_x'].dt.day
    df['weekday'] = df['dt_x'].dt.weekday
    print("Extracted date features", flush=True)
    
    def parse_wait_time(x):
        try:
            h, m, s = map(int, x.split(':'))
            return h * 60 + m + s / 60
        except: return 0
    df['waiting_time_min'] = df['waiting_time'].apply(parse_wait_time)
    print("Parsed wait times", flush=True)
    
    le_priority = LabelEncoder()
    df['priority_encoded'] = le_priority.fit_transform(df['priority'])
    joblib.dump(le_priority, os.path.join(MODELS_DIR, 'label_encoder_priority.pkl'))
    print("Encoded priority", flush=True)
    
    le_risk = LabelEncoder()
    df['risk_encoded'] = le_risk.fit_transform(df['risk'])
    joblib.dump(le_risk, os.path.join(MODELS_DIR, 'label_encoder_risk.pkl'))
    print("Encoded risk", flush=True)
    
    df['patient_load_ratio'] = df['waiting_patients'] / (df['examined_patients'] + 1)
    df['queue_congestion'] = df['waiting_patients'] * (df['priority_encoded'] + 1)
    print("Calculated ratios", flush=True)
    
    # Drop unnecessary columns
    df = df.drop(['timestamp', 'dt_x', 'dt_y', 'date_only', 'waiting_time', 'priority', 'risk', 'emergency_room'], axis=1)
    df = df.dropna()
    print(f"Final data shape: {df.shape}", flush=True)
    
    return df

def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train():
    df = load_and_clean_data()
    
    X = df.drop('waiting_time_min', axis=1)
    y = df['waiting_time_min']
    
    # Save feature names for inference
    joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, 'feature_names.pkl'))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    
    model = build_model(X_train_scaled.shape[1])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("Training model...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluation
    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"\nModel Evaluation:")
    print(f"MAE: {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")
    
    # Save model
    model.save(os.path.join(MODELS_DIR, 'trained_ann_regression.h5'))
    print(f"Model saved to {MODELS_DIR}")
    
    # Plot history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss (MSE)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Model MAE')
    plt.legend()
    
    plt.savefig(os.path.join(MODELS_DIR, 'training_history.png'))
    print("Training history plot saved.")

if __name__ == '__main__':
    train()
