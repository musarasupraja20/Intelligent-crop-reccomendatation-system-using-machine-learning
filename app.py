from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load fresh models
model = pickle.load(open('models/crop_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
le = pickle.load(open('models/label_encoder.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictor')
def predictor():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Access by name to ensure correct order
        inputs = [
            float(request.form.get('N')),
            float(request.form.get('P')),
            float(request.form.get('K')),
            float(request.form.get('temp')),
            float(request.form.get('hum')),
            float(request.form.get('ph')),
            float(request.form.get('rain'))
        ]
        
        # Scale input
        final_input = scaler.transform(np.array([inputs]))
        
        # Predict
        prediction = model.predict(final_input)
        crop = le.inverse_transform(prediction)[0]
        
        return render_template('predict.html', prediction_text=crop, input_data=inputs)
    except Exception as e:
        return render_template('predict.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)