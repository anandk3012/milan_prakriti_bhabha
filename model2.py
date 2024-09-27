import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from gensim.models import Word2Vec
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')



# Load dataset
df = pd.read_csv('my_flask_app/water_usage_data.csv')
df = df.drop(['District'], axis=1)

# Prepare sentences for Word2Vec
sentences = df[['Crop', 'Irrigation Method']].values.tolist()

# Train Word2Vec model
model_w = Word2Vec(sentences, vector_size=10, window=3, min_count=1, workers=4)

# Save the trained Word2Vec model
model_w.save('my_flask_app/word2vec_model.bin')

# Neural network model
def modelNN(X_train):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

# Get vector representation for word
def get_vector(word, model):
    try:
        return model.wv[word]  # Try to get the word vector
    except KeyError:
        print(f"Word '{word}' not found in the model. Returning zero vector.")
        return np.zeros(model.vector_size)  # Return zero vector if word not found


# Convert input data into feature vectors
def convert_input(crop, irrigation, water_availability):
    crop_vector = get_vector(crop, model_w)
    irrigation_vector = get_vector(irrigation, model_w)
    X = pd.DataFrame([np.append(crop_vector, irrigation_vector)], columns=[f"Feature_{i}" for i in range(len(crop_vector) + len(irrigation_vector))])
    X['Water Availability (liters/hectare)'] = water_availability
    return X

# Prepare input data for model
X = pd.concat([convert_input(crop, irrigation, availability) for crop, irrigation, availability in 
               zip(df['Crop'], df['Irrigation Method'], df['Water Availability (liters/hectare)'])])

y = df['Water Consumption (liters/hectare)']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = modelNN(X_train)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('my_flask_app/water_usage_model.h5')  # Save the Keras model
