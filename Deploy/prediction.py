import tensorflow as tf
import numpy as np
import pandas as pd
import random
from io import BytesIO
import urllib.request
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Read Cleaned Data
df = pd.read_csv("final_data.csv")
print(len(df))

# Global Variables
EMBEDDING_DIM = 100
MAXLEN = 16
TRUNCATING = 'post'
PADDING = 'post'
OOV_TOKEN = "<OOV>"
MAX_EXAMPLES = len(df)
TRAINING_SPLIT = 0.9


def random_sampling():
    random.seed(42)
    # Get the indices of the DataFrame
    indices = df.index.tolist()
    # Perform random sampling on the indices
    selected_indices = random.sample(indices, MAX_EXAMPLES)
    # Select the corresponding sentences and labels based on the sampled indices
    sentences = df.loc[selected_indices, 'reviews']
    labels = df.loc[selected_indices, 'sentiment']
    return sentences, labels

# Declare sentences and labels variable
sentences, labels = random_sampling()

def train_val_split(sentences, labels, training_split): 
    train_size = int(len(sentences)*training_split)

    train_sentences = sentences[:train_size]
    train_labels = labels[:train_size]

    test_sentences = sentences[train_size:]
    test_labels = labels[train_size:]
    
    return train_sentences, test_sentences, train_labels, test_labels

# Declare all the train & test labels and sentences
train_sentences, test_sentences, train_labels, test_labels = train_val_split(sentences, labels, TRAINING_SPLIT)

def fit_tokenizer(sentences, oov_token):
    tokenizer = Tokenizer(oov_token = OOV_TOKEN)
    tokenizer.fit_on_texts(sentences) 
    return tokenizer

# Declare tokenizer
tokenizer = fit_tokenizer(train_sentences, OOV_TOKEN)

# Load Model
model = tf.keras.models.load_model('model_1.h5')

# Random Text as an Input
random_text = "The lectures for this subject were average. The lecturer was competent in delivering the content, but it lacked the wow factor. It was neither exceptional nor disappointing"

def model_predict(random_text):
    random_text_sequence = tokenizer.texts_to_sequences([random_text])
    random_text_sequence = pad_sequences(random_text_sequence, maxlen = MAXLEN, padding=PADDING, truncating =TRUNCATING)
    prediction = model.predict(random_text_sequence)[0]
    predicted_label = np.argmax(prediction)
    sentiment = "Positive" if predicted_label == 2 else "Neutral" if predicted_label == 1 else "Negative"
    return sentiment, prediction

result, s= model_predict(random_text)
print(result)
print(s)
print("================================")