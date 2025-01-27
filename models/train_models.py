import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import pickle
import re

def clean_text(text):
    """Enhanced text cleaning with minimal processing"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s-]', '', text)  # Keep hyphens
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text.strip()

def main():
    # Load dataset with header in first row
    df = pd.read_csv('data/Grievance-dataset.csv', encoding='ISO-8859-1')
    
    # Create combined category using available partial information
    df['Combined_Category'] = df.apply(
        lambda row: f"{row['Category']}::{row['Sub-category']}" 
        if not pd.isna(row['Category']) or not pd.isna(row['Sub-category']) 
        else np.nan,
        axis=1
    )

    # Identify labeled rows (any category/sub-category present)
    labeled_mask = df['Combined_Category'].notna()
    labeled_data = df[labeled_mask].copy()
    unlabeled_data = df[~labeled_mask].copy()

    print(f"Labeled samples available: {len(labeled_data)}")
    print(f"Unlabeled samples to predict: {len(unlabeled_data)}")

    if len(labeled_data) == 0:
        raise ValueError("No labeled data available. Check Category/Sub-category columns.")

    # Enhanced text cleaning
    labeled_data['clean_text'] = labeled_data['Grievance Description'].apply(clean_text)
    unlabeled_data['clean_text'] = unlabeled_data['Grievance Description'].apply(clean_text)

    # Create model pipeline with adjusted parameters
    model = make_pipeline(
        TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95,
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'  # Only words with 3+ letters
        ),
        RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
    )

    # Train model
    try:
        model.fit(labeled_data['clean_text'], labeled_data['Combined_Category'])
    except ValueError as e:
        print("Error during training:", e)
        print("Sample clean texts:", labeled_data['clean_text'].head().tolist())
        return

    # Predict categories for unlabeled data
    if len(unlabeled_data) > 0:
        predictions = model.predict(unlabeled_data['clean_text'])
        df.loc[~labeled_mask, 'Combined_Category'] = predictions

    # Split combined category back into original columns
    def split_category(row):
        if pd.isna(row['Combined_Category']):
            return 'General', 'Uncategorized'
        parts = str(row['Combined_Category']).split('::')
        return (
            parts[0] if len(parts) > 0 else 'General',
            parts[1] if len(parts) > 1 else 'Uncategorized'
        )

    df[['Category', 'Sub-category']] = df.apply(
        lambda row: split_category(row),
        axis=1,
        result_type='expand'
    )

    # Urgency mapping with railway-specific rules
    URGENCY_KEYWORDS = {
        'High': ['emergency', 'medical', 'accident', 'fire', 'derailment'],
        'Medium': ['delay', 'cancellation', 'reservation', 'refund', 'ticket'],
        'Low': ['cleanliness', 'information', 'feedback', 'staff', 'facility']
    }

    def assign_urgency(text):
        text = clean_text(text)
        for level, keywords in URGENCY_KEYWORDS.items():
            if any(f' {kw} ' in f' {text} ' for kw in keywords):
                return level
        return 'Medium' #Default

    df['Urgency Level'] = df['Grievance Description'].apply(assign_urgency)

    # Save results
    df.to_csv('data/Updated_Grievance-dataset.csv', index=False)
    with open('models/category_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("\nTraining completed successfully!")
    print("Category Distribution:")
    print(df['Category'].value_counts())
    print("\nUrgency Level Distribution:")
    print(df['Urgency Level'].value_counts())

if __name__ == "__main__":
    main()