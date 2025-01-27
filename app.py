from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import pickle
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os

# Download NLTK data
nltk.download('vader_lexicon')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///grievance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Load ML models
complaint_model = pickle.load(open('models/complaint_classifier.pkl', 'rb'))
urgency_model = pickle.load(open('models/urgency_classifier.pkl', 'rb'))

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'staff', 'admin'
    department = db.Column(db.String(50))

class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pnr = db.Column(db.String(10), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    priority = db.Column(db.Integer)
    status = db.Column(db.String(20), default='submitted')
    department = db.Column(db.String(50))
    staff_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    feedback = db.Column(db.Text)
    sentiment = db.Column(db.String(20))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit_complaint', methods=['POST'])
def submit_complaint():
    pnr = request.form['pnr']
    description = request.form['description']
    
    # Predict category using ML model
    category = predict_category(description)
    
    # Predict urgency
    priority = predict_urgency(description)
    
    # Create complaint
    complaint = Complaint(
        pnr=pnr,
        description=description,
        category=category,
        priority=priority
    )
    db.session.add(complaint)
    db.session.commit()
    
    return jsonify({
        'complaint_id': complaint.id,
        'category': category
    })

@app.route('/track_complaint/<complaint_id>')
def track_complaint(complaint_id):
    complaint = Complaint.query.get_or_404(complaint_id)
    return render_template('track_complaint.html', complaint=complaint)

@app.route('/staff_dashboard')
@login_required
def staff_dashboard():
    if current_user.role != 'staff':
        return redirect(url_for('home'))
    complaints = Complaint.query.filter_by(staff_id=current_user.id).all()
    return render_template('staff_dashboard.html', complaints=complaints)

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        return redirect(url_for('home'))
    department = current_user.department
    complaints = Complaint.query.filter_by(department=department).all()
    staff = User.query.filter_by(role='staff', department=department).all()
    return render_template('admin_dashboard.html', complaints=complaints, staff=staff)

# Helper functions
def predict_category(description):
    # Use the loaded complaint_model to predict category
    # This is a placeholder - implement actual prediction logic
    return "Cleanliness"

def predict_urgency(description):
    # Use the loaded urgency_model to predict priority
    # This is a placeholder - implement actual prediction logic
    return 2

def analyze_sentiment(feedback):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(feedback)
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)