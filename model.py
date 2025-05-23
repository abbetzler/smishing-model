import sys
import pandas as pd
import re
from io import StringIO
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from textstat import flesch_reading_ease
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input, Concatenate, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
from features import FeatureExtractor


# Download the stopwords and punkt resources (if not already downloaded)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

dataset_csv = "datasets/analysisdataset.csv"
dataset_txt = "datasets/SMSSmishCollection.txt"
dataset_pt = "datasets/sms_dataset_pt.txt"
columns_to_keep = ["MainText", "Phishing"]


def read_dataset():
    print("### Reading Dataset ###")
    # Attempt to read the CSV file with encoding handling
    try:
        df = pd.read_csv(dataset_csv, encoding="utf-8")  # Default encoding
        print("Dataset loaded successfully with UTF-8 encoding!")
    except UnicodeDecodeError:
        print("UTF-8 encoding failed. Retrying with ISO-8859-1 encoding...")
        try:
            df = pd.read_csv(dataset_csv, encoding="ISO-8859-1")  # Fallback encoding
            print("Dataset loaded successfully with ISO-8859-1 encoding!")
        except Exception as e:
            print(f"Failed to load the dataset. Error: {e}")
            exit()

    # Check if specified columns exist in the dataset
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    if missing_columns:
        print(f"Error: The following columns are not in the dataset: {missing_columns}")
        exit()

    return df


def clean_dataset(dataframe):
    print("### Cleaning Dataset ###")
    # Select only the specified columns
    cleaned_df = dataframe[columns_to_keep].copy()

    # Use .loc to safely assign the new column
    cleaned_df.loc[:, "Label"] = cleaned_df["Phishing"].apply(assign_label)

    # Remove the "Phishing" column
    cleaned_df = cleaned_df.drop(columns=["Phishing"])

    # Clean SMS and add to DataFrame
    cleaned_df.loc[:, "CleanText"] = cleaned_df["MainText"].apply(clean_sms)

    # Parse txt file
    parsed_data = []

    with open(dataset_txt, 'r', encoding='utf-8') as file:
        for line in file:
            # Split the line into label and text
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                label, text = parts
                parsed_data.append({'MainText': text, 'Label': label})

    # Create a new DataFrame from the parsed data
    txt_df = pd.DataFrame(parsed_data)
    txt_df.loc[:, "CleanText"] = txt_df["MainText"].apply(clean_sms, language="en")

    # Dataset in Portuguese
    parsed_data_pt = []

    with open(dataset_pt, 'r', encoding='utf-8') as file:
        for line in file:
            # Split the line into label and text
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                label, text = parts
                parsed_data_pt.append({'MainText': text, 'Label': label})

    txt_df_pt = pd.DataFrame(parsed_data_pt)
    txt_df_pt.loc[:, "CleanText"] = txt_df_pt["MainText"].apply(clean_sms, language="pt")

    # Concatenate the existing DataFrame with the new DataFrame
    combined_df = pd.concat([cleaned_df, txt_df, txt_df_pt], ignore_index=True)

    # Save the combined dataset to a new CSV file
    output_file_path = "datasets/combined_dataset.csv"
    combined_df.to_csv(output_file_path, index=False)
    print(f"Combined dataset saved to: {output_file_path}")

    return combined_df


def assign_label(phishing_value):
    if pd.isna(phishing_value) or phishing_value == 0:
        return "ham"
    elif phishing_value > 0:
        return "smish"
    else:
        return "unknown"


def clean_sms(text, language="en"):
    # Convert text to lower case
    text = text.lower()

    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    filtered_text = ""

    # Use nltk to remove stop words
    if language == "en":
        words = word_tokenize(text, 'english')
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_text = ' '.join(filtered_words)
    elif language == "pt":
        words = word_tokenize(text, 'portuguese')
        stop_words = set(stopwords.words('portuguese'))
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_text = ' '.join(filtered_words)

    return filtered_text


def ml_model(dataset):
    print("### Building Model ###")

    # Preprocess text data
    X_text = dataset['CleanText'].astype(str).values
    y = dataset['Label'].values

    # Encode labels (smish -> 1, ham -> 0)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Tokenization
    max_words = 5000  # Vocabulary size
    max_len = 100  # Max length of SMS messages

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_text)
    X_sequences = tokenizer.texts_to_sequences(X_text)
    X_padded = pad_sequences(X_sequences, maxlen=max_len, padding='post')

    # Extract additional features
    fe = FeatureExtractor()
    X_features = np.array([fe.extract_features(text) for text in X_text])

    # Normalize numerical features
    scaler = StandardScaler()
    X_features[:, 3:] = scaler.fit_transform(X_features[:, 3:])

    # TF-IDF Transformation
    vectorizer = TfidfVectorizer(max_features=1000)  # Limit TF-IDF features to 1000
    X_tfidf = vectorizer.fit_transform(X_text).toarray()

    # Split dataset
    X_train_text, X_test_text, X_train_features, X_test_features, X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
        X_padded, X_features, X_tfidf, y, test_size=0.2, random_state=42
    )

    text_input = Input(shape=(max_len,))
    embedding = Embedding(max_words, 128)(text_input)
    lstm = LSTM(64, return_sequences=True)(embedding)
    lstm = Dropout(0.5)(lstm)
    lstm = LSTM(32)(lstm)

    features_input = Input(shape=(4,))
    features_dense = Dense(16, activation="relu")(features_input)

    # TF-IDF input
    tfidf_input = Input(shape=(1000,))
    tfidf_dense = Dense(16, activation="relu")(tfidf_input)

    # Concatenate LSTM output with additional features
    concatenated = Concatenate()([lstm, features_dense, tfidf_dense])
    output = Dense(1, activation="sigmoid")(concatenated)

    # Compile the model
    model = Model(inputs=[text_input, features_input, tfidf_input], outputs=output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model
    model.fit(
        [X_train_text, X_train_features, X_train_tfidf], y_train,
        epochs=5, batch_size=32, validation_data=([X_test_text, X_test_features, X_test_tfidf], y_test)
    )

    # Save model
    model.save("lstm_sms_classifier.keras")

    # Save tokenizer for later use
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model saved successfully!")

    y_pred_probs = model.predict([X_test_text, X_test_features, X_test_tfidf])
    y_pred = (y_pred_probs > 0.5).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


if __name__ == "__main__":
    df = read_dataset()
    cleaned_dataset = clean_dataset(df)
    ml_model(cleaned_dataset)