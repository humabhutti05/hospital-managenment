import numpy as np

CONGESTION_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}
CONGESTION_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}

ARRIVAL_RATES = {0: 2, 1: 5, 2: 10}   # patients per ABM step

def congestion_label(code):
    return CONGESTION_LABELS.get(int(code), 'Unknown')

def congestion_color(code):
    return CONGESTION_COLORS.get(int(code), '#95a5a6')

def arrival_rate(code):
    return ARRIVAL_RATES.get(int(code), 2)
