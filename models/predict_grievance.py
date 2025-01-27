import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Define a function to clean the text (same as in the model training script)
def clean_text(text):
    """Enhanced text cleaning with minimal processing"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s-]', '', text)  # Keep hyphens
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text.strip()

# Load the trained model (the model was saved as 'category_model.pkl')
with open('models/category_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the urgency mapping and function
URGENCY_KEYWORDS = {
    'High': ['emergency', 'medical', 'accident', 'fire', 'derailment'],
    'Medium': ['delay', 'cancellation', 'reservation', 'refund', 'ticket'],
    'Low': ['cleanliness', 'information', 'feedback', 'staff', 'facility']
}

def assign_urgency(text):
    """Assign urgency level based on keywords."""
    text = clean_text(text)
    for level, keywords in URGENCY_KEYWORDS.items():
        if any(f' {kw} ' in f' {text} ' for kw in keywords):
            return level
    return 'Medium'  # Default if no keywords match

# Function to predict categories and urgency levels
def predict_grievance(file_path):
    # Load the new complaints data
    new_complaints = pd.read_csv(file_path)

    # Clean the grievance descriptions
    new_complaints['clean_text'] = new_complaints['Grievance Description'].apply(clean_text)

    # Predict categories for the new complaints
    predictions = model.predict(new_complaints['clean_text'])
    new_complaints['Predicted Category'] = predictions

    # Assign urgency levels to the new complaints
    new_complaints['Urgency Level'] = new_complaints['Grievance Description'].apply(assign_urgency)

    # Save the predictions to a new CSV file
    new_complaints.to_csv('data/Predicted_Grievances.csv', index=False)

    # Return the updated dataframe
    return new_complaints[['Grievance Description', 'Predicted Category', 'Urgency Level']]

# Run the prediction function
if __name__ == "__main__":
    file_path = 'data/new_complaints.csv'  # Path to the new complaints CSV file
    predictions_df = predict_grievance(file_path)
    
    # Optionally print the results or save it
    print(predictions_df.head())
