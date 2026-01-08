from flask import Flask, render_template, request, session, jsonify
import numpy as np
import joblib
from models.logistic_model import LogisticRegression
from models.model_evaluation import ModelEvaluator, load_sample_data
import json
from datetime import datetime
__main__.LogisticRegression = LogisticRegression

app = Flask(__name__)
app.secret_key = "cvd_secret_key_123"

@app.context_processor
def inject_config():
    return {
        'current_year': datetime.now().year,
        'now': datetime.now()
    }

# Load models
model = joblib.load('models/cvd_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Initialize model evaluator (load and cache results on startup)
try:
    X, y = load_sample_data()
    evaluator = ModelEvaluator(X, y)
    evaluator.train_models()
    evaluator.kfold_validation(k=5)
    
    model_comparison = evaluator.get_model_comparison()
    cv_results = evaluator.get_cv_results()
except Exception as e:
    print(f"Error loading evaluation data: {e}")
    model_comparison = {}
    cv_results = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")
    
    try:
        # Get data
        form = request.form
        name = form["name"]
        X = np.array([[float(form["age"]), int(form["gender"]), float(form["height"]),
                      float(form["weight"]), float(form["bp_hi"]), float(form["bp_lo"]),
                      int(form["cholesterol"]), int(form["gluc"]), int(form["smoke"]),
                      int(form["alco"]), int(form["active"])]])
        
        # Prediction
        X_scaled = scaler.transform(X)
        proba = round(model.predict_proba(X_scaled)[0, 1] * 100, 1)
        
        # Logic
        risk_level = "High Risk" if proba >= 70 else "Moderate Risk" if proba >= 30 else "Low Risk"
        risk_color = "#e76f51" if proba >= 70 else "#e9c46a" if proba >= 30 else "#2a9d8f"
        
        # Calculate BMI
        bmi = round(float(form["weight"]) / (float(form["height"]) / 100) ** 2, 2)
        
        # Save to History (Session)
        if 'history' not in session:
            session['history'] = []
        
        new_entry = {
            "name": name,
            "proba": proba,
            "level": risk_level,
            "color": risk_color,
            "age": form["age"],
            "gender": form["gender"],
            "height": form["height"],
            "weight": form["weight"],
            "bp_hi": form["bp_hi"],
            "bp_lo": form["bp_lo"],
            "cholesterol": form["cholesterol"],
            "gluc": form["gluc"],
            "smoke": form["smoke"],
            "alco": form["alco"],
            "active": form["active"],
            "bmi": bmi,
        }
        
        history = [new_entry] + session['history']
        session['history'] = history[:5]  # Keep last 5
        session.modified = True
        
        return render_template("result.html",
                              name=name,
                              proba=proba,
                              risk_level=risk_level,
                              risk_color=risk_color,
                              bmi=bmi,
                              inputs=new_entry,
                              history=session['history'])
    
    except Exception as e:
        return f"Error: {e}"

@app.route("/models")
def models_comparison():
    """Display model comparison page"""
    return render_template("models.html", model_data=model_comparison, cv_data=cv_results)

@app.route("/api/models")
def api_models():
    """API endpoint for model data"""
    return jsonify(model_comparison)

@app.route("/crossvalidation")
def cross_validation():
    """Display cross-validation results page"""
    return render_template("crossvalidation.html", cv_data=cv_results)

@app.route("/api/crossvalidation")
def api_cross_validation():
    """API endpoint for cross-validation data"""
    return jsonify(cv_results)

if __name__ == "__main__":
    app.run(debug=True)
