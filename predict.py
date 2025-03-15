import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from features import FeatureExtractor

# Load the model
model = load_model("lstm_sms_classifier.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# Function to preprocess input SMS for prediction
def preprocess_sms(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post')

    # Extract additional features
    fe = FeatureExtractor()
    features = np.array([fe.extract_features(text)])

    # Normalize numerical features
    features[:, 3:] = scaler.transform(features[:, 3:])

    # Convert text to Bag-of-Words vector
    bow_vector = vectorizer.transform([text]).toarray()

    return padded, features, bow_vector


# Predict function
def predict_sms(text):
    X_text, X_features, X_bow = preprocess_sms(text)
    prediction = model.predict([X_text, X_features, X_bow])[0, 0]
    print(prediction)
    return "smish" if prediction > 0.5 else "ham"


if __name__ == "__main__":
    # Example SMS
    smish = ("WARNING:(Criminal Investigation Division) I.R.S is filing a lawsuit against you, for more information "
             "call +17038798780 on urgent basis, Otherwise your arrest warrant will be forwarded to your local police "
             "department and your property and bank accounts and social benifits will be frozen by government.")

    ham = "Hey, are we still meeting for coffee at 3?"

    print(predict_sms(smish))
    print(predict_sms(ham))
