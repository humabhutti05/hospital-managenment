import pandas as pd
import numpy as np

def load_and_preprocess(att_path, er_path):
    att = pd.read_csv(att_path)
    er  = pd.read_csv(er_path)

    # Join with ER data to get hospital_name for better labeling
    er_names = er[['emergency_room', 'hospital_name', 'is_pediatric']].copy()
    
    # Translate Priority Colors
    color_map = {'Rosso': 'Red', 'Giallo': 'Yellow', 'Verde': 'Green', 'Bianco': 'White'}
    att['priority'] = att['priority'].map(color_map).fillna(att['priority'])

    # Helper to clean and translate hospital names
    def clean_hospital_name(row):
        h_name = str(row['hospital_name'])
        is_ped = row['is_pediatric']
        
        # Remove common Italian prefixes
        h_name = h_name.replace('Ospedale di ', '').replace('Ospedale ', '')
        h_name = h_name.replace('Presidio Ospedaliero ', '').replace('IRCCS ', '')
        h_name = h_name.replace('di ', '').strip(' "')
        
        # Handle some specific long names
        if 'Santa Maria della Misericordia' in h_name:
            h_name = 'Santa Maria Hospital'
            
        suffix = "Pediatric ER" if is_ped else "ER"
        return f"{h_name} {suffix}"

    # We need to clean names in both dataframes
    er_names['clean_name'] = er_names.apply(clean_hospital_name, axis=1)
    
    # Map the clean name back to attendance data
    name_map = er_names.set_index('emergency_room')['clean_name'].to_dict()
    att['emergency_room_clean'] = att['emergency_room'].map(name_map)
    
    # Update er dataframe for return
    er['emergency_room'] = er['emergency_room'].map(name_map)
    att['emergency_room'] = att['emergency_room_clean']
    att.drop(columns=['emergency_room_clean'], inplace=True)

    att['datetime']       = pd.to_datetime(att['timestamp'], unit='ms')
    att['hour']           = att['datetime'].dt.hour
    att['day_of_week']    = att['datetime'].dt.dayofweek
    att['month']          = att['datetime'].dt.month

    def parse_wait(t):
        try:
            parts = str(t).split(':')
            return int(parts[0]) * 60 + int(parts[1])
        except:
            return 0

    att['waiting_time_min'] = att['waiting_time'].apply(parse_wait)
    att.drop(columns=['timestamp', 'datetime', 'waiting_time'], inplace=True)

    er_slim = er[['emergency_room', 'is_pediatric']].copy()
    df = att.merge(er_slim, on='emergency_room', how='left')

    priority_map = {'Red': 4, 'Yellow': 3, 'Green': 2, 'White': 1}
    df['priority_num'] = df['priority'].map(priority_map).fillna(1).astype(int)
    df['is_pediatric'] = df['is_pediatric'].fillna(0).astype(int)
    df.drop(columns=['emergency_room', 'priority'], inplace=True)
    df.dropna(inplace=True)

    def label(x):
        if x == 0:   return 0
        elif x <= 2: return 1
        else:        return 2

    df['congestion'] = df['waiting_patients'].apply(label)

    print(f"Dataset shape: {df.shape}")
    print(f"Congestion distribution:\n{df['congestion'].value_counts().sort_index()}")
    print(f"Thresholds => Low:0  Medium:1-2  High:3+")

    return df, 0, 2
