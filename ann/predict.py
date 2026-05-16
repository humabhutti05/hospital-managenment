import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import joblib
import tensorflow as tf

BASE = os.path.dirname(os.path.dirname(__file__))

_model  = None
_scaler = None

def _load():
    global _model, _scaler
    if _model is None:
        _model  = tf.keras.models.load_model(os.path.join(BASE, 'models', 'trained_ann.h5'))
        _scaler = joblib.load(os.path.join(BASE, 'models', 'scaler.pkl'))

def predict_congestion(examined, waiting, wait_min, hour, dow, month, priority_num, is_pediatric):
    """Return (congestion_code, confidence) where code 0=Low 1=Medium 2=High."""
    _load()
    X = np.array([[examined, waiting, wait_min, hour, dow, month, priority_num, is_pediatric]])
    X_scaled = _scaler.transform(X)
    probs = _model.predict(X_scaled, verbose=0)[0]
    code  = int(np.argmax(probs))
    conf  = float(probs[code])
    return code, conf

def predict_from_row(row_dict):
    return predict_congestion(
        row_dict.get('examined_patients', 0),
        row_dict.get('waiting_patients', 0),
        row_dict.get('waiting_time_min', 0),
        row_dict.get('hour', 12),
        row_dict.get('day_of_week', 0),
        row_dict.get('month', 1),
        row_dict.get('priority_num', 2),
        row_dict.get('is_pediatric', 0),
    )
