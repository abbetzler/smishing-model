import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from features import FeatureExtractor
from fastapi import FastAPI
from pydantic import BaseModel

# Load the model
model = load_model("lstm_sms_classifier.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Define the FastAPI app
app = FastAPI()


# Define the request model
class SMSInput(BaseModel):
    message: str


# Function to preprocess input SMS for prediction
def preprocess_sms(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post')

    # Extract additional features
    fe = FeatureExtractor()
    features = np.array([fe.extract_features(text)])

    # Normalize numerical features
    features[:, 3:] = scaler.transform(features[:, 3:])

    # Convert text to TF-IDF vector
    tfidf_vector = vectorizer.transform([text]).toarray()

    return padded, features, tfidf_vector


# Prediction endpoint
@app.post("/predict")
async def predict_sms(data: SMSInput):
    X_text, X_features, X_tfidf = preprocess_sms(data.message)
    prediction = model.predict([X_text, X_features, X_tfidf])[0, 0]
    print(prediction)
    return {"message": data.message, "prediction": str(prediction)}


# Health check
@app.get("/")
def root():
    return {"status": "SMS classifier API is running"}
