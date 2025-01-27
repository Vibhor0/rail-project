from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import pickle
import joblib  # Changed from pickle for better compatibility
import numpy as np
import re
import os
from pathlib import Path
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download NLTK data
nltk.download('vader_lexicon')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///grievance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Load ML model and vectorizer
MODEL_PATH = Path(__file__).parent / 'models/category_model.pkl'
try:
    category_pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    category_pipeline = None

# Database Models (updated with proper fields)
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='staff')  # 'staff', 'admin'
    department = db.Column(db.String(50))

class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pnr = db.Column(db.String(10), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    sub_category = db.Column(db.String(50))  # Added sub-category
    urgency = db.Column(db.String(20), nullable=False)  # Changed from priority
    status = db.Column(db.String(20), default='submitted')
    department = db.Column(db.String(50))
    staff_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    feedback = db.Column(db.Text)
    sentiment = db.Column(db.String(20))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Text cleaning function (must match training preprocessing)
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# Prediction functions
URGENCY_KEYWORDS = {
    'High': ['emergency', 'medical', 'accident', 'fire','derailment', 'security', 'unsafe', 'threat', 'danger', 'hazard', 'terrorist', 'violence', 'attack', 'crisis', 'critical', 'injury', 'death', 'disaster', 'fatal','bomb', 'rescue', 'evacuation', 'explosion', 'armed', 'distress'],
    'Medium': ['delay', 'cancellation', 'reservation', 'refund', 'ticket', 'train delay', 'missed', 'late', 'unavailability', 'partial', 'inconvenience', 'boarding', 'rebooking', 'service', 'transport', 'connections','food', 'meal', 'service delay', 'reception', 'staff response', 'seat allocation'],
    'Low': ['cleanliness', 'information', 'feedback', 'staff', 'facility', 'comfort', 'service quality', 'restroom', 'water', 'AC', 'maintenance', 'non-functioning', 'luggage', 'seating', 'noise', 'light', 'temperature', 'washroom', 'toilet', 'hygiene', 'staff attitude', 'wait time', 'communication', 'delay explanation', 'atmosphere']
}

def predict_category(text):
    if not category_pipeline:
        return "General", "Uncategorized"
    
    try:
        cleaned_text = clean_text(text)
        prediction = category_pipeline.predict([cleaned_text])[0]
        if '::' in prediction:
            return prediction.split('::')
        return prediction, 'General'
    except Exception as e:
        print(f"Prediction error: {e}")
        return "General", "Uncategorized"

def predict_urgency(text):
    cleaned = clean_text(text)
    for level, keywords in URGENCY_KEYWORDS.items():
        if any(kw in cleaned for kw in keywords):
            return level
    return 'Medium'

# Updated routes
@app.route('/submit_complaint', methods=['POST'])
def submit_complaint():
    pnr = request.form['pnr']
    description = request.form['description']
    
    # Get predictions
    category, subcat = predict_category(description)
    urgency = predict_urgency(description)
    
    # Create complaint
    complaint = Complaint(
        pnr=pnr,
        description=description,
        category=category,
        sub_category=subcat,
        urgency=urgency
    )
    
    db.session.add(complaint)
    db.session.commit()
    
    return jsonify({
        'complaint_id': complaint.id,
        'category': f"{category} ({subcat})",
        'urgency': urgency
    })

@app.route('/staff_dashboard')
@login_required
def staff_dashboard():
    if current_user.role != 'staff':
        return redirect(url_for('home'))
    
    complaints = Complaint.query.filter_by(staff_id=current_user.id).all()
    return render_template('staff_dashboard.html', 
                         complaints=complaints,
                         urgency_colors={
                             'High': 'danger',
                             'Medium': 'warning',
                             'Low': 'success'
                         })

@app.route('/update_status/<int:complaint_id>', methods=['POST'])
@login_required
def update_status(complaint_id):
    complaint = Complaint.query.get_or_404(complaint_id)
    new_status = request.form.get('status')
    
    if new_status in ['in_progress', 'resolved']:
        complaint.status = new_status
        if new_status == 'resolved' and request.form.get('feedback'):
            complaint.feedback = request.form.get('feedback')
            complaint.sentiment = analyze_sentiment(complaint.feedback)
        
        db.session.commit()
        flash('Status updated successfully', 'success')
    else:
        flash('Invalid status', 'danger')
    
    return redirect(url_for('staff_dashboard'))

def analyze_sentiment(feedback):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(feedback)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    return 'Neutral'

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Create default admin user if not exists
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                password='adminpassword',  # In production, hash this!
                role='admin',
                department='Management'
            )
            db.session.add(admin)
            db.session.commit()
    app.run(debug=True)