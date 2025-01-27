# models/p_g.py
import pandas as pd
import pickle
import re
from pathlib import Path

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

def predict_urgency(text):
    urgency_keywords = {
        'High': ['emergency', 'medical', 'accident', 'fire','derailment', 'security', 'unsafe', 'threat', 'danger', 'hazard', 'terrorist', 'violence', 'attack', 'crisis', 'critical', 'injury', 'death', 'disaster', 'fatal','bomb', 'rescue', 'evacuation', 'explosion', 'armed', 'distress'],
        'Medium': ['delay', 'cancellation', 'reservation', 'refund', 'ticket', 'train delay', 'missed', 'late', 'unavailability', 'partial', 'inconvenience', 'boarding', 'rebooking', 'service', 'transport', 'connections','food', 'meal', 'service delay', 'reception', 'staff response', 'seat allocation'],
        'Low': ['cleanliness', 'information', 'feedback', 'staff', 'facility', 'comfort', 'service quality', 'restroom', 'water', 'AC', 'maintenance', 'non-functioning', 'luggage', 'seating', 'noise', 'light', 'temperature', 'washroom', 'toilet', 'hygiene', 'staff attitude', 'wait time', 'communication', 'delay explanation', 'atmosphere']
    }
    cleaned = clean_text(text)
    for level, keywords in urgency_keywords.items():
        if any(kw in cleaned for kw in keywords):
            return level
    return 'Medium'

def predict_grievances(input_path):
    model_path = Path(__file__).parent / 'category_model.pkl'
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    new_data = pd.read_csv(input_path, encoding='ISO-8859-1')
    new_data['clean_text'] = new_data['Grievance Description'].apply(clean_text)
    predictions = pipeline.predict(new_data['clean_text'])
    
    results = []
    for desc, pred in zip(new_data['Grievance Description'], predictions):
        category, subcat = pred.split('::') if '::' in pred else (pred, 'General')
        urgency = predict_urgency(desc)
        results.append({
            'description': desc,
            'category': category,
            'subcategory': subcat,
            'urgency': urgency
        })
    
    return results