import mesa
import random

class PatientAgent(mesa.Agent):
    def __init__(self, model, priority_label, arrival_time, patience_threshold=60):
        super().__init__(model)
        self.priority_label = priority_label # Rosso, Giallo, Verde, Bianco
        self.arrival_time = arrival_time
        self.patience_threshold = patience_threshold
        self.assigned_hospital = None
        self.predicted_wait = 0
        self.actual_wait = 0
        self.state = 'waiting' # waiting, in_treatment, done, dropped
        self.treatment_time = random.randint(5, 15) # Default treatment time

        
    def step(self):
        if self.state == 'waiting':
            self.actual_wait += 1
            if self.actual_wait > self.patience_threshold:
                self.state = 'dropped'
        elif self.state == 'in_treatment':
            self.treatment_time -= 1
            if self.treatment_time <= 0:
                self.state = 'done'

class DoctorAgent(mesa.Agent):
    def __init__(self, model, doctor_id, specialization='General'):
        super().__init__(model)
        self.doctor_id = doctor_id
        self.specialization = specialization
        self.available = True
        self.current_patient = None
        self.patients_treated = 0
        self.efficiency = 1.0
        
    def step(self):
        if self.current_patient:
            # Decrease patient's treatment time based on doctor's efficiency
            # To simulate fatigue, if efficiency < 1.0, the decrement could be smaller,
            # effectively increasing the time it takes to treat.
            # However, patient agent decrements treatment_time in its own step.
            # A simpler way: when assigning a patient, adjust their treatment_time 
            # by dividing by efficiency. See route_patient or assignment in model.py.
            if self.current_patient.state == 'done':
                self.available = True
                self.current_patient = None
                self.patients_treated += 1
                
                fatigue_mult = getattr(self.model, 'fatigue_multiplier', 0.05)
                self.efficiency = max(0.5, 1.0 - (self.patients_treated * fatigue_mult))
        
        if self.available:
            pass
