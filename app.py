from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Change this part in the load_models() function
def load_models():
    try:
        # Using joblib instead of pickle
        svm_model = joblib.load('svm_model.pkl')
        lr_model = joblib.load('lr_model.pkl')
        return svm_model, lr_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

svm_model, lr_model = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        try:
            # Get input features from the form
            age = int(request.form['age'])
            job = request.form['job']
            marital = request.form['marital']
            education = request.form['education']
            default = request.form['default']
            housing = request.form['housing']
            loan = request.form['loan']
            
            # Additional fields required for prediction (with defaults)
            balance = int(request.form.get('balance', 0))
            contact = request.form.get('contact', 'unknown')
            day = int(request.form.get('day', 1))
            month = request.form.get('month', 'jan')
            campaign = int(request.form.get('campaign', 1))
            pdays = int(request.form.get('pdays', 999))
            previous = int(request.form.get('previous', 0))
            poutcome = request.form.get('poutcome', 'unknown')
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'job': [job],
                'marital': [marital],
                'education': [education],
                'default': [default],
                'balance': [balance],
                'housing': [housing],
                'loan': [loan],
                'contact': [contact],
                'day': [day],
                'month': [month],
                'campaign': [campaign],
                'pdays': [pdays],
                'previous': [previous],
                'poutcome': [poutcome]
            })
            
            # Make predictions
            svm_prediction = svm_model.predict(input_data)
            lr_prediction = lr_model.predict(input_data)
            
            # Format predictions
            svm_result = 'Yes' if svm_prediction[0] == 1 else 'No'
            lr_result = 'Yes' if lr_prediction[0] == 1 else 'No'
            
            return render_template('result.html', 
                                  svm_prediction=svm_result,
                                  lr_prediction=lr_result,
                                  input_data=input_data.to_dict('records')[0])
        
        except Exception as e:
            return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [int(data.get('age', 30))],
            'job': [data.get('job', 'unknown')],
            'marital': [data.get('marital', 'unknown')],
            'education': [data.get('education', 'unknown')],
            'default': [data.get('default', 'no')],
            'balance': [int(data.get('balance', 0))],
            'housing': [data.get('housing', 'no')],
            'loan': [data.get('loan', 'no')],
            'contact': [data.get('contact', 'unknown')],
            'day': [int(data.get('day', 1))],
            'month': [data.get('month', 'jan')],
            'campaign': [int(data.get('campaign', 1))],
            'pdays': [int(data.get('pdays', 999))],
            'previous': [int(data.get('previous', 0))],
            'poutcome': [data.get('poutcome', 'unknown')]
        })
        
        # Make predictions
        svm_prediction = svm_model.predict(input_data)
        lr_prediction = lr_model.predict(input_data)
        
        # Return JSON response
        return jsonify({
            'svm_prediction': int(svm_prediction[0]),
            'svm_result': 'Yes' if svm_prediction[0] == 1 else 'No',
            'lr_prediction': int(lr_prediction[0]),
            'lr_result': 'Yes' if lr_prediction[0] == 1 else 'No',
            'input_data': input_data.to_dict('records')[0]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)