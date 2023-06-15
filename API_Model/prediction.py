import tensorflow as tf
import numpy as np
from modelML import *

# Load Model
mdl = tf.keras.models.load_model('model.h5')
# Random Text as an Input
random_text = "Smart , but can't teach"

def model_predict(random_text):
    random_text_sequence = tokenizer.texts_to_sequences([random_text])
    random_text_sequence = pad_sequences(random_text_sequence, maxlen = MAXLEN, padding=PADDING, truncating =TRUNCATING)
    prediction = mdl.predict(random_text_sequence)[0]
    predicted_label = np.argmax(prediction)
    sentiment = "Positive" if predicted_label == 2 else "Neutral" if predicted_label == 1 else "Negative"
    return sentiment