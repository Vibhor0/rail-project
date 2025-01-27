import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import pickle
import re

def clean_text(text):
    """Basic text cleaning function"""
    if pd.isna(text):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def main():
    # Load dataset with proper headers
    df = pd.read_csv('data/Grievance-dataset.csv', encoding='ISO-8859-1')
    
    # Keep original columns but focus on key fields
    print(f"Initial dataset shape: {df.shape}")
    print(f"Columns with data: {df.count().sort_values()}")

    # Create combined category for first 77 rows
    df['Combined_Category'] = np.where(
        (df['Category'].notna()) & (df['Sub-category'].notna()),
        df['Category'] + "::" + df['Sub-category'], 
        np.nan
    )

    # Split data into labeled (first 77 rows) and unlabeled
    labeled_mask = (df['Category'].notna()) & (df['Sub-category'].notna())
    labeled_data = df[labeled_mask].copy()
    unlabeled_data = df[~labeled_mask].copy()

    print(f"\nLabeled samples available: {len(labeled_data)}")
    print(f"Unlabeled samples to predict: {len(unlabeled_data)}")

    # Clean text data
    labeled_data['clean_text'] = labeled_data['Grievance Description'].apply(clean_text)
    unlabeled_data['clean_text'] = unlabeled_data['Grievance Description'].apply(clean_text)

    # Create and train model pipeline
    model = make_pipeline(
        TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        ),
        RandomForestClassifier(
            n_estimators=150,
            class_weight='balanced',
            random_state=42
        )
    )

    # Train on labeled data
    X_train = labeled_data['clean_text']
    y_train = labeled_data['Combined_Category']
    model.fit(X_train, y_train)

    # Predict categories for unlabeled data
    X_predict = unlabeled_data['clean_text']
    predicted_categories = model.predict(X_predict)
    
    # Update combined category for entire dataset
    df.loc[~labeled_mask, 'Combined_Category'] = predicted_categories

    # Split combined category back into original columns
    def split_category(row):
        if pd.isna(row['Combined_Category']):
            return 'Unknown', 'Unknown'
        parts = row['Combined_Category'].split("::")
        return parts[0], parts[1] if len(parts) > 1 else 'General'

    df[['Category', 'Sub-category']] = df.apply(
        lambda row: split_category(row),
        axis=1,
        result_type='expand'
    )

    # Define urgency mapping rules
    URGENCY_RULES = {
        'high': ['medical', 'emergency', 'accident', 'fire', 'safety'],
        'medium': ['food', 'water', 'ticket', 'payment', 'reservation'],
        'low': ['cleanliness', 'information', 'feedback', 'staff']
    }

    def assign_urgency(row):
        combined = (row['Category'] + ' ' + row['Sub-category']).lower()
        for level, keywords in URGENCY_RULES.items():
            if any(kw in combined for kw in keywords):
                return level.capitalize()
        return 'Medium'  # Default case

    df['Urgency Level'] = df.apply(assign_urgency, axis=1)

    # Save results
    df.to_csv('data/Updated_Grievance-dataset.csv', index=False)
    with open('models/category_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("\nTraining completed successfully!")
    print("Final category distribution:")
    print(df['Category'].value_counts())
    print("\nUrgency level distribution:")
    print(df['Urgency Level'].value_counts())

if __name__ == "__main__":
    main()