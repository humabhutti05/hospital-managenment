import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import joblib

from utils.preprocessing import load_and_preprocess

BASE = os.path.dirname(os.path.dirname(__file__))

def train():
    print("=== ANN Training ===")
    df, q33, q66 = load_and_preprocess(
        os.path.join(BASE, 'data', 'attendances.csv'),
        os.path.join(BASE, 'data', 'emergencyRooms.csv')
    )

    FEATURES = ['examined_patients', 'waiting_patients', 'waiting_time_min',
                'hour', 'day_of_week', 'month', 'priority_num', 'is_pediatric']
    X = df[FEATURES].values
    y = df['congestion'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Ensure models directory exists
    models_dir = os.path.join(BASE, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    np.save(os.path.join(models_dir, 'thresholds.npy'), [q33, q66])

    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5,
        batch_size=256,
        class_weight=class_weight,
        callbacks=[es],
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {acc*100:.2f}%")

    model.save(os.path.join(models_dir, 'trained_ann.h5'))
    print(f"Model saved to models/trained_ann.h5")

    return history, acc

if __name__ == '__main__':
    train()
