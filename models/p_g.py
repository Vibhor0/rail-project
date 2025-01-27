import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    """Text cleaning (must match training preprocessing exactly)"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

def split_category(combined):
    """Mirror the category splitting from training"""
    if pd.isna(combined):
        return 'General', 'Uncategorized'
    parts = str(combined).split('::')
    return (
        parts[0] if len(parts) > 0 else 'General',
        parts[1] if len(parts) > 1 else 'Uncategorized'
    )

# Load the trained pipeline (now handles both vectorizer and model)
with open('models/category_model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

URGENCY_KEYWORDS = {
    'High': ['emergency', 'medical', 'accident', 'fire', 'derailment'],
    'Medium': ['delay', 'cancellation', 'reservation', 'refund', 'ticket'],
    'Low': ['cleanliness', 'information', 'feedback', 'staff', 'facility']
}

def predict_grievance(file_path):
    # Load new data with proper error handling
    try:
        new_data = pd.read_csv(file_path, encoding='ISO-8859-1')
    except FileNotFoundError:
        raise ValueError(f"File not found at {file_path}")
    
    # Clean text using same preprocessing
    new_data['clean_text'] = new_data['Grievance Description'].apply(clean_text)
    
    # Predict combined categories
    combined_preds = pipeline.predict(new_data['clean_text'])
    
    # Split into Category/Sub-category columns
    new_data[['Category', 'Sub-category']] = [
        split_category(pred) for pred in combined_preds
    ]
    
    # Assign urgency (modified for better keyword matching)
    def assign_urgency(text):
        cleaned = clean_text(text)
        for level, keywords in URGENCY_KEYWORDS.items():
            if any(kw in cleaned for kw in keywords):
                return level
        return 'Medium'
    
    new_data['Urgency Level'] = new_data['Grievance Description'].apply(assign_urgency)
    
    # Save while preserving original structure
    new_data.to_csv('data/Predicted_Grievances.csv', index=False)
    
    return new_data[['Grievance Description', 'Category', 'Sub-category', 'Urgency Level']]

if __name__ == "__main__":
    input_path = 'data/new_complaints.csv'  # Ensure this file exists
    results = predict_grievance(input_path)
    print("Prediction Sample:")
    print(results.head(3))