import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load the trained models
with open('models/complaint_classifier.pkl', 'rb') as f:
    complaint_model = pickle.load(f)

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Read the new complaints from CSV
new_complaints = pd.read_csv('data/new_complaints.csv', encoding='ISO-8859-1')

# Preprocess the complaints (if there are missing descriptions, drop them)
new_complaints.dropna(subset=['Grievance Description'], inplace=True)

# Step 1: Transform the complaints using the trained TF-IDF vectorizer
X_new_complaints_tfidf = tfidf.transform(new_complaints['Grievance Description'])

# Debug: Check the shape of the transformed data and some sample data
print(f"TF-IDF Transformation Shape: {X_new_complaints_tfidf.shape}")
print(f"Sample of Transformed Data (First 2 complaints):")
print(X_new_complaints_tfidf[:2].toarray())

# Step 2: Predict the categories for the new complaints
predicted_categories = complaint_model.predict(X_new_complaints_tfidf)

# Debug: Check the first few predictions
print(f"Predicted Categories (first 5): {predicted_categories[:5]}")

# Assign predicted categories to the new complaints
new_complaints['Predicted Category'] = predicted_categories

# Step 3: Apply heuristic for Urgency Level based on Predicted Category
def assign_urgency(row):
    category = row['Predicted Category'].lower()
    
    # Assign urgency based on the predicted category (you can refine this based on actual use cases)
    if 'safety' in category or 'health' in category:
        return 'High'
    elif 'train' in category or 'service' in category:
        return 'Medium'
    else:
        return 'Low'

# Apply the heuristic to the new complaints
new_complaints['Urgency Level'] = new_complaints.apply(assign_urgency, axis=1)

# Debug: Check the results before saving
print("Updated Complaints with Predicted Category and Urgency Level:")
print(new_complaints[['Grievance Description', 'Predicted Category', 'Urgency Level']].head())

# Save the new complaints with predictions and urgency levels
new_complaints.to_csv('data/predicted_complaints.csv', index=False)

print("Predictions for new complaints have been saved to 'data/predicted_complaints.csv'.")
