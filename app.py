from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# =============================================
# Load ML Model & Scaler
# =============================================
model_path = os.path.join('ml_model', 'model.pkl')
scaler_path = os.path.join('ml_model', 'scaler.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

print("✅ Model & Scaler loaded successfully!")

# Crop info dictionary
crop_info = {
    'rice': {'season': 'Kharif', 'water': 'High', 'temp': '20-35°C', 'emoji': '🌾'},
    'wheat': {'season': 'Rabi', 'water': 'Medium', 'temp': '12-25°C', 'emoji': '🌿'},
    'maize': {'season': 'Kharif', 'water': 'Medium', 'temp': '18-27°C', 'emoji': '🌽'},
    'chickpea': {'season': 'Rabi', 'water': 'Low', 'temp': '10-25°C', 'emoji': '🫘'},
    'kidneybeans': {'season': 'Kharif', 'water': 'Medium', 'temp': '18-24°C', 'emoji': '🫘'},
    'pigeonpeas': {'season': 'Kharif', 'water': 'Low', 'temp': '18-29°C', 'emoji': '🌱'},
    'mothbeans': {'season': 'Kharif', 'water': 'Low', 'temp': '25-35°C', 'emoji': '🌱'},
    'mungbean': {'season': 'Kharif', 'water': 'Low', 'temp': '25-35°C', 'emoji': '🫘'},
    'blackgram': {'season': 'Kharif', 'water': 'Low', 'temp': '25-35°C', 'emoji': '🫘'},
    'lentil': {'season': 'Rabi', 'water': 'Low', 'temp': '15-25°C', 'emoji': '🫘'},
    'pomegranate': {'season': 'Annual', 'water': 'Low', 'temp': '25-35°C', 'emoji': '🍎'},
    'banana': {'season': 'Annual', 'water': 'High', 'temp': '20-35°C', 'emoji': '🍌'},
    'mango': {'season': 'Summer', 'water': 'Medium', 'temp': '24-27°C', 'emoji': '🥭'},
    'grapes': {'season': 'Annual', 'water': 'Medium', 'temp': '15-40°C', 'emoji': '🍇'},
    'watermelon': {'season': 'Summer', 'water': 'High', 'temp': '22-30°C', 'emoji': '🍉'},
    'muskmelon': {'season': 'Summer', 'water': 'Medium', 'temp': '25-35°C', 'emoji': '🍈'},
    'apple': {'season': 'Winter', 'water': 'Medium', 'temp': '21-24°C', 'emoji': '🍎'},
    'orange': {'season': 'Winter', 'water': 'Medium', 'temp': '13-37°C', 'emoji': '🍊'},
    'papaya': {'season': 'Annual', 'water': 'Medium', 'temp': '22-26°C', 'emoji': '🧡'},
    'coconut': {'season': 'Annual', 'water': 'High', 'temp': '27-30°C', 'emoji': '🥥'},
    'cotton': {'season': 'Kharif', 'water': 'Medium', 'temp': '21-30°C', 'emoji': '🌸'},
    'jute': {'season': 'Kharif', 'water': 'High', 'temp': '24-35°C', 'emoji': '🌿'},
    'coffee': {'season': 'Annual', 'water': 'High', 'temp': '15-28°C', 'emoji': '☕'},
}

# =============================================
# ROUTES
# =============================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Input values
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        # Validate inputs
        if not (0 <= N <= 140): return jsonify({'error': 'N value 0-140 hona chahiye'}), 400
        if not (0 <= P <= 145): return jsonify({'error': 'P value 0-145 hona chahiye'}), 400
        if not (0 <= K <= 205): return jsonify({'error': 'K value 0-205 hona chahiye'}), 400
        if not (0 <= temperature <= 50): return jsonify({'error': 'Temperature 0-50°C hona chahiye'}), 400
        if not (0 <= humidity <= 100): return jsonify({'error': 'Humidity 0-100% hona chahiye'}), 400
        if not (0 <= ph <= 14): return jsonify({'error': 'pH 0-14 hona chahiye'}), 400
        if not (0 <= rainfall <= 300): return jsonify({'error': 'Rainfall 0-300mm hona chahiye'}), 400

        # Predict
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled).max() * 100

        # Get crop info
        info = crop_info.get(prediction.lower(), {
            'season': 'N/A', 'water': 'N/A', 'temp': 'N/A', 'emoji': '🌱'
        })

        return jsonify({
            'success': True,
            'crop': prediction,
            'confidence': round(confidence, 2),
            'season': info['season'],
            'water_requirement': info['water'],
            'temperature_range': info['temp'],
            'emoji': info['emoji']
        })

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK', 'message': 'AVANI API is running! 🌱'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)