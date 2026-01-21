"""
House Price Prediction Web Application
Student: Wesley Iluobe
Matric Number: 23CH034193

Flask web application that loads the trained model and provides
a user interface for predicting house prices.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
MODEL_PATH = 'model/house_price_model.pkl'
SCALER_PATH = 'model/scaler.pkl'

# Check if model files exist
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully!")
else:
    print("Warning: Model files not found. Please run model_building.py first.")
    model = None
    scaler = None

# Feature names used in training
FEATURE_NAMES = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt']


@app.route('/')
def home():
    """Render the home page with the prediction form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Check if model is loaded
        if model is None or scaler is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please ensure the model files exist.'
            })
        
        # Get form data
        overall_qual = int(request.form.get('overall_qual', 5))
        gr_liv_area = float(request.form.get('gr_liv_area', 1500))
        total_bsmt_sf = float(request.form.get('total_bsmt_sf', 1000))
        garage_cars = int(request.form.get('garage_cars', 2))
        full_bath = int(request.form.get('full_bath', 2))
        year_built = int(request.form.get('year_built', 2000))
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'OverallQual': [overall_qual],
            'GrLivArea': [gr_liv_area],
            'TotalBsmtSF': [total_bsmt_sf],
            'GarageCars': [garage_cars],
            'FullBath': [full_bath],
            'YearBuilt': [year_built]
        })
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Format the prediction
        formatted_price = "${:,.2f}".format(prediction)
        
        return jsonify({
            'success': True,
            'prediction': formatted_price,
            'raw_prediction': float(prediction),
            'features': {
                'Overall Quality': overall_qual,
                'Living Area (sq ft)': gr_liv_area,
                'Basement Area (sq ft)': total_bsmt_sf,
                'Garage Capacity': garage_cars,
                'Full Bathrooms': full_bath,
                'Year Built': year_built
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON predictions."""
    try:
        if model is None or scaler is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded.'
            }), 500
        
        data = request.get_json()
        
        input_data = pd.DataFrame({
            'OverallQual': [data.get('overall_qual', 5)],
            'GrLivArea': [data.get('gr_liv_area', 1500)],
            'TotalBsmtSF': [data.get('total_bsmt_sf', 1000)],
            'GarageCars': [data.get('garage_cars', 2)],
            'FullBath': [data.get('full_bath', 2)],
            'YearBuilt': [data.get('year_built', 2000)]
        })
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'formatted': "${:,.2f}".format(prediction)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
