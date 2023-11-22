import string
import demoji
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

ps = PorterStemmer()

def process_message(msg):
    # Preprocess the text data
    special_chars = string.punctuation
    
    stop_words = set(stopwords.words('english'))
    
    text = str(msg).lower()
    text = demoji.replace(text, '')
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{special_chars}]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [ps.stem(word) for word in tokens]
    
    return ' '.join(tokens)


def load_and_predict(  msg):
    # Load the saved model
    model = load_model('SimpleRNN.keras')

    # Load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
     tokenizer = pickle.load(handle)

    # Preprocess the message
    processed_message = process_message(msg)
    
    # Use word_index from the tokenizer configuration
    sequences = [[tokenizer.word_index.get(word, 0) for word in processed_message.split()]]
    
    padded_sequences = pad_sequences(sequences, maxlen=model.input_shape[1])

    # Predict the class
    prediction = model.predict(padded_sequences)
    predicted_class = np.argmax(prediction)


    # Map the predicted class to the corresponding label
    labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_label = labels[predicted_class]

    print("Predicted Label:", predicted_label)

    return predicted_label

