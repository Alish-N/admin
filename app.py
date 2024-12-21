from flask import Flask, request, jsonify, render_template, make_response, send_from_directory
from flask_cors import CORS
from model_preprocessing import ISPCustomerSatisfaction
import pandas as pd
import traceback
import os

app = Flask(__name__, static_url_path='')
# Enable CORS for all routes
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

# Initialize and load the model
model = ISPCustomerSatisfaction()

# Check if model file exists, if not, train the model
if not os.path.exists('isp_satisfaction_model.joblib'):
    try:
        print("Training new model...")
        metrics = model.train_model('isp_admin_data.csv')
        model.save_model('isp_satisfaction_model.joblib')
        print("Model trained successfully!")
        print("\nModel Performance Metrics:")
        print("Training Metrics:")
        print(f"Regression - R²: {metrics['train_metrics']['regression']['r2']:.4f}")
        print(f"Classification - Accuracy: {metrics['train_metrics']['classification']['accuracy']:.4f}")
        print("\nTest Metrics:")
        print(f"Regression - R²: {metrics['test_metrics']['regression']['r2']:.4f}")
        print(f"Classification - Accuracy: {metrics['test_metrics']['classification']['accuracy']:.4f}")
    except Exception as e:
        print(f"Error training model: {str(e)}")
else:
    try:
        print("Loading existing model...")
        model.load_model('isp_satisfaction_model.joblib')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        # Get input data
        data = request.get_json()
        if not data:
            response = make_response(
                jsonify({
                    'status': 'error',
                    'message': 'No input data provided'
                }),
                400
            )
            response.headers['Content-Type'] = 'application/json'
            return response
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Make prediction
        result = model.predict(input_df)
        
        response = make_response(
            jsonify({
                'status': 'success',
                'prediction': float(result['prediction']),
                'feature_importance': {
                    k: float(v) for k, v in result['feature_importance'].items()
                },
                'message': 'Prediction successful'
            })
        )
        response.headers['Content-Type'] = 'application/json'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    
    except Exception as e:
        # Get the full error traceback
        error_traceback = traceback.format_exc()
        print(f"Prediction error: {error_traceback}")
        
        response = make_response(
            jsonify({
                'status': 'error',
                'message': str(e),
                'details': error_traceback
            }),
            500
        )
        response.headers['Content-Type'] = 'application/json'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

@app.route('/train', methods=['POST', 'OPTIONS'])
def train():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        # Train model
        metrics = model.train_model('isp_admin_data.csv')
        
        # Save the model after successful training
        model.save_model('isp_satisfaction_model.joblib')
        
        # Convert numpy values to native Python types
        def convert_metrics(d):
            if isinstance(d, dict):
                return {k: convert_metrics(v) for k, v in d.items()}
            elif hasattr(d, 'dtype'):  # numpy type
                return float(d)
            elif isinstance(d, (list, tuple)):
                return [convert_metrics(i) for i in d]
            return d

        formatted_metrics = convert_metrics(metrics)
        
        response = make_response(
            jsonify({
                'status': 'success',
                'metrics': formatted_metrics,
                'message': 'Model trained and saved successfully'
            })
        )
        response.headers['Content-Type'] = 'application/json'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Training error: {error_traceback}")
        
        response = make_response(
            jsonify({
                'status': 'error',
                'message': str(e),
                'details': error_traceback
            }),
            500
        )
        response.headers['Content-Type'] = 'application/json'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 