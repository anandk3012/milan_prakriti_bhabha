from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from flask_cors import CORS
import tensorflow
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import joblib
import nltk
nltk.download('punkt')

app = Flask(__name__)
CORS(app)

# Load the trained models
model_w = Word2Vec.load('word2vec_model.bin')  # Load your trained Word2Vec model
water_usage_model = load_model('water_usage_model.h5')  # Load Keras model correctly
irrigation_model = joblib.load('irrigation_model.pkl')

# Load and prepare Word2Vec model

# Convert input data into feature vectors
def get_vector(word, model):
    try:
        return model.wv[word]  # Try to get the word vector
    except KeyError:
        print(f"Word '{word}' not found in the model. Returning zero vector.")
        return np.zeros(model.vector_size)  # Return zero vector if word not found


def convert_input(crop, irrigation, water_availability):
    crop_vector = get_vector(crop, model_w)
    irrigation_vector = get_vector(irrigation, model_w)
    X = pd.DataFrame([np.append(crop_vector, irrigation_vector)], columns=[f"Feature_{i}" for i in range(2)])
    X['Water Availability (liters/hectare)'] = water_availability
    return X

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/recommendation', methods=['POST'])
def recommendation():
    data = request.json

    irrigation_data = pd.DataFrame({
        'SoilMoisture': [data['soilMoisture']],
        'temperature': [data['temperature']],
        'Humidity': [data['humidity']]
    })

    irrigation_prediction = irrigation_model.predict(irrigation_data)[0]

    response = {
        'recommendation': 'Good' if irrigation_prediction == 1 else 'Bad'
    }

    return jsonify(response)

@app.route('/api/amount', methods=['POST'])
def amount():
    data = request.json

    water_usage_data = convert_input(data['cropName'], data['irrigationType'], data['waterAvailability'])

    water_usage_prediction = water_usage_model.predict(water_usage_data)[0][0]  # Get the predicted value


    response = {
        'amount': water_usage_prediction
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
