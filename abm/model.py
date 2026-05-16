import mesa
import pandas as pd
import numpy as np
import random
from abm.agents import PatientAgent, DoctorAgent
import joblib
import os

class HospitalModel(mesa.Model):
    def __init__(self, hospital_data_path, ann_model_path=None, scaler_path=None, 
                 arrival_rate=0.5, num_doctors_per_hosp=3, covid_risk='low risk',
                 fatigue_multiplier=0.05, patience_threshold=60):
        super().__init__()
        self.schedule = mesa.time.RandomActivation(self)
        self.arrival_rate = arrival_rate
        self.covid_risk = covid_risk
        self.hospitals_df = pd.read_csv(hospital_data_path)
        self.num_doctors_per_hosp = num_doctors_per_hosp
        self.fatigue_multiplier = fatigue_multiplier
        self.patience_threshold = patience_threshold
        self.total_dropped = 0
        
        # Load ANN components if available
        self.ann_model = None
        if ann_model_path and os.path.exists(ann_model_path):
            import tensorflow as tf
            self.ann_model = tf.keras.models.load_model(ann_model_path, compile=False)
            self.scaler = joblib.load(scaler_path)
            self.le_priority = joblib.load(os.path.join(os.path.dirname(scaler_path), 'label_encoder_priority.pkl'))
            self.le_risk = joblib.load(os.path.join(os.path.dirname(scaler_path), 'label_encoder_risk.pkl'))
            self.feature_names = joblib.load(os.path.join(os.path.dirname(scaler_path), 'feature_names.pkl'))

        # Setup hospitals and doctors
        self.hospital_queues = {name: [] for name in self.hospitals_df['emergency_room']}
        self.hospital_doctors = {name: [] for name in self.hospitals_df['emergency_room']}
        
        for name in self.hospitals_df['emergency_room']:
            for i in range(self.num_doctors_per_hosp):
                d = DoctorAgent(self, f"{name}_D{i}")
                d.hospital = name
                self.schedule.add(d)
                self.hospital_doctors[name].append(d)
                
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={
                "Total Patients": lambda m: len([a for a in m.schedule.agents if isinstance(a, PatientAgent)]),
                "Waiting Patients": lambda m: len([a for a in m.schedule.agents if isinstance(a, PatientAgent) and a.state == 'waiting']),
                "Active Treatments": lambda m: len([a for a in m.schedule.agents if isinstance(a, PatientAgent) and a.state == 'in_treatment']),
                "Completed": lambda m: len([a for a in m.schedule.agents if isinstance(a, PatientAgent) and a.state == 'done']),
                "Patients Lost": lambda m: m.total_dropped,
                "Avg Wait Time": lambda m: np.mean([a.actual_wait for a in m.schedule.agents if isinstance(a, PatientAgent) and a.state in ['done', 'in_treatment']] or [0])
            }
        )

    def predict_wait_time(self, hospital_name, priority):
        if not self.ann_model:
            return len(self.hospital_queues[hospital_name]) * 10 # Dummy heuristic
        
        # Prepare features for ANN
        # Note: In a real simulation, we'd use current hour/day/etc.
        import datetime
        now = datetime.datetime.now()
        
        # Mocking current hospital stats
        waiting_patients = len(self.hospital_queues[hospital_name])
        examined_patients = 10 # Mock historical context
        
        # Priority encoding
        try:
            p_enc = self.le_priority.transform([priority])[0]
        except:
            p_enc = 0
            
        # Risk encoding
        try:
            r_enc = self.le_risk.transform([self.covid_risk])[0]
        except:
            r_enc = 0

        # Feature vector matching training script
        features = {
            'is_pediatric': self.hospitals_df[self.hospitals_df['emergency_room'] == hospital_name]['is_pediatric'].values[0],
            'risk_encoded': r_enc,
            'hour': now.hour,
            'day': now.day,
            'weekday': now.weekday(),
            'priority_encoded': p_enc,
            'examined_patients': examined_patients,
            'waiting_patients': waiting_patients,
            'patient_load_ratio': waiting_patients / (examined_patients + 1),
            'queue_congestion': waiting_patients * (p_enc + 1)
        }
        
        # Reorder to match training
        feat_vals = [features[col] for col in self.feature_names if col in features]
        X = np.array([feat_vals])
        X_scaled = self.scaler.transform(X)
        
        prediction = self.ann_model.predict(X_scaled, verbose=0)
        return max(0, prediction[0][0])

    def route_patient(self, patient):
        best_hospital = None
        min_wait = float('inf')
        
        for name in self.hospital_queues.keys():
            pred_wait = self.predict_wait_time(name, patient.priority_label)
            if pred_wait < min_wait:
                min_wait = pred_wait
                best_hospital = name
        
        patient.assigned_hospital = best_hospital
        patient.predicted_wait = min_wait
        self.hospital_queues[best_hospital].append(patient)

    def step(self):
        # 1. New patient arrival
        # Arrival rate adjusted by COVID risk
        risk_multiplier = 1.0
        if 'high' in self.covid_risk: risk_multiplier = 2.0
        elif 'medium' in self.covid_risk: risk_multiplier = 1.5
        
        if random.random() < (self.arrival_rate * risk_multiplier):
            priority = random.choices(['Rosso', 'Giallo', 'Verde', 'Bianco'], weights=[0.1, 0.3, 0.4, 0.2])[0]
            p = PatientAgent(self, priority, self.schedule.steps, patience_threshold=self.patience_threshold)
            self.schedule.add(p)
            self.route_patient(p)
            
        # 2. Assign waiting patients to available doctors
        for hosp_name, queue in self.hospital_queues.items():
            # Remove dropped patients from queue
            dropped_this_step = [p for p in queue if p.state == 'dropped']
            self.total_dropped += len(dropped_this_step)
            queue[:] = [p for p in queue if p.state != 'dropped']
            
            # Sort queue by priority: Rosso(3) > Giallo(2) > Verde(1) > Bianco(0)
            # (Assuming LabelEncoder roughly follows this or we define mapping)
            queue.sort(key=lambda x: x.priority_label, reverse=True) # Simple alpha sort for now
            
            available_docs = [d for d in self.hospital_doctors[hosp_name] if d.available]
            for doc in available_docs:
                if queue:
                    p = queue.pop(0)
                    p.state = 'in_treatment'
                    p.treatment_time = int(p.treatment_time / doc.efficiency) # Adjust for fatigue
                    doc.available = False
                    doc.current_patient = p
        
        self.datacollector.collect(self)
        self.schedule.step()
