import sys
import pandas as pd
import re
from io import StringIO
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textstat import flesch_reading_ease
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import keras


# Download the stopwords and punkt resources (if not already downloaded)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

dataset_csv = "datasets/analysisdataset.csv"
dataset_txt = "datasets/SMSSmishCollection.txt"
columns_to_keep = ["MainText", "Phishing"]

# List of common suspicious words in smishing messages
SUSPICIOUS_WORDS = [
    "urgent", "immediately", "act now", "attention", "important",
    "limited time", "alert", "hurry", "final notice", "winner",
    "claim now", "call now", "verify", "action required", "reply now",
    "warning", "IRS", "I.R.S", "arrest", "account", "free", "bank",
    "click", "claim", "limited", "lottery"
]

# Initialize TF-IDF and BoW
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
bow_vectorizer = CountVectorizer(max_features=5000)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
#model = TFMobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=2)
tinybert_model = TFAutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", from_pt=True)


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
    txt_df.loc[:, "CleanText"] = txt_df["MainText"].apply(clean_sms)

    # Concatenate the existing DataFrame with the new DataFrame
    combined_df = pd.concat([cleaned_df, txt_df], ignore_index=True)

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


def clean_sms(text):
    # Convert text to lower case
    text = text.lower()

    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Use nltk to remove stop words
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)

    return filtered_text


def extract_features(dataset):
    print("### Extracting Features ###")

    # Convert labels to int (0 for 'ham', 1 for 'smish')
    converted_labels = dataset['Label'].apply(lambda x: 1 if x == 'smish' else 0).values

    contains_phone = np.array(dataset['CleanText'].apply(contains_phone_number)).reshape(-1, 1)
    contains_suspicious = np.array(dataset['CleanText'].apply(contains_suspicious_words)).reshape(-1, 1)
    readability_score = np.array(dataset['CleanText'].apply(flesch_reading_ease)).reshape(-1, 1)
    noun_count = np.array(dataset['CleanText'].apply(count_nouns)).reshape(-1, 1)

    features = np.hstack((contains_phone, contains_suspicious, readability_score, noun_count))
    labels = np.array(converted_labels)

    return features, labels


def contains_phone_number(text):
    """Check if the SMS contains a phone number"""
    # Regular expression pattern to match phone numbers
    phone_pattern = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{1,4}\)[-.\s]?)?(?:\d{1,4}[-.\s]?){2,4}\d\b')

    # Find all phone numbers in the text
    phone_numbers = phone_pattern.findall(text)

    return int(bool(phone_numbers))


def contains_suspicious_words(text):
    """Check if an SMS contains suspicious words."""
    text = text.lower()
    for word in SUSPICIOUS_WORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', text):  # Match whole words
            return 1
    return 0


def count_nouns(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    return sum(1 for word, pos in tagged_words if pos.startswith('NN'))


def vectorize_text(texts, fit=False):
    """Convert SMS text into TF-IDF & BoW vectors"""
    if fit:
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts).toarray()
        bow_matrix = bow_vectorizer.fit_transform(texts).toarray()
    else:
        tfidf_matrix = tfidf_vectorizer.transform(texts).toarray()
        bow_matrix = bow_vectorizer.transform(texts).toarray()

    return np.hstack((tfidf_matrix, bow_matrix))


def ml_model(dataset):
    print("### Building Model ###")

    # Tokenize and prepare input for MobileBERT
    encoded_input = tokenizer(dataset['CleanText'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="tf")

    # Extract input tensors
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]

    print(input_ids.shape, attention_mask.shape)

    # Forward pass through TinyBERT
    outputs = tinybert_model(input_ids=input_ids, attention_mask=attention_mask)

    # Extract embeddings
    embeddings = outputs.last_hidden_state

    print(embeddings.shape)

    # Convert labels to int (0 for 'ham', 1 for 'smish')
    converted_labels = dataset['Label'].apply(lambda x: 1 if x == 'smish' else 0).values

    # Convert labels to TensorFlow tensors
    labels = tf.convert_to_tensor(converted_labels, dtype=tf.int32)

    '''
    custom_features, labels = extract_features(dataset)
    text_vectors = vectorize_text(dataset['CleanText'].tolist(), fit=True)

    # Define explicit MobileBERT inputs
    input_ids = Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(128,), dtype=tf.int32, name="attention_mask")

    # Combine all features
    final_inputs = {
        "input_ids": bert_inputs["input_ids"],
        "attention_mask": bert_inputs["attention_mask"],
        "custom_features": tf.convert_to_tensor(custom_features, dtype=tf.float32),
        "text_vectors": tf.convert_to_tensor(text_vectors, dtype=tf.float32)
    }

    # Use the custom embedding layer
    bert_embeddings = MobileBertEmbedLayer(mobilebert)([input_ids, attention_mask])
    pooled_output = GlobalAveragePooling1D()(bert_embeddings)  # Convert embeddings to fixed size

    # Define custom inputs
    custom_features_input = Input(shape=(4,), dtype=tf.float32, name="custom_features")
    text_vectors_input = Input(shape=(5000 * 2,), name="text_vectors")  # TF-IDF + BoW

    # Concatenate features
    merged = Concatenate()([pooled_output, custom_features_input, text_vectors_input])

    # Fully Connected Layers
    x = Dense(128, activation="relu")(merged)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)

    # Classification layer
    output = Dense(1, activation="sigmoid", name="output")(x)

    # Build final model
    model = Model(inputs=[input_ids, attention_mask, custom_features_input, text_vectors_input], outputs=output)

    '''

    # Create model instance
    model = TinyBERTClassifier(tinybert_model)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=5e-5), loss="binary_crossentropy", metrics=["accuracy"])

    # Create a dataset
    batch_size = 16
    dataset = tf.data.Dataset.from_tensor_slices(((input_ids, attention_mask), labels)).batch(batch_size)

    # Train the model
    model.fit(dataset, epochs=3)

    return model


def save_model(pretrained_model):
    # Save the model
    pretrained_model.export("tinybert_smishing_model")

    # Convert the SavedModel to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model("tinybert_smishing_model")

    # Enable optimizations (optional for mobile efficiency)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert the model
    tflite_model = converter.convert()

    # Save the TFLite model
    with open("tinybert_smishing_model.tflite", "wb") as f:
        f.write(tflite_model)

    print("TFLite model saved successfully!")


# Define custom classification model
@keras.saving.register_keras_serializable()
class TinyBERTClassifier(Model):
    def __init__(self, tiny_model, dropout_rate=0.3, **kwargs):
        super(TinyBERTClassifier, self).__init__()
        self.tinybert = tiny_model
        self.dropout = Dropout(dropout_rate)
        self.classifier = Dense(1, activation="sigmoid")

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.tinybert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled_output)
        return self.classifier(x)

    def get_config(self):
        config = super().get_config()
        config.update({"dropout_rate": self.dropout.rate})
        return config

    @classmethod
    def from_config(cls, config):
        tb_model = TFAutoModel.from_pretrained(
            "huawei-noah/TinyBERT_General_4L_312D", from_pt=True
        )
        return cls(tiny_model=tb_model, **config)


if __name__ == "__main__":
    df = read_dataset()
    cleaned_dataset = clean_dataset(df)
    ml_model = ml_model(cleaned_dataset)
    save_model(ml_model)

    # Example predictions
    smish = ("WARNING:(Criminal Investigation Division) I.R.S is filing a lawsuit against you, for more information "
             "call +17038798780 on urgent basis, Otherwise your arrest warrant will be forwarded to your local police "
             "department and your property and bank accounts and social benifits will be frozen by government.")

    ham = "Hey, are we still meeting for coffee at 3?"

    #print(predict_sms(smish, lstm_model))
    #print(predict_sms(ham, lstm_model))
