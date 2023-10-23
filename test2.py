import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from nltk.corpus import stopwords
import string
import demoji
import re
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Suponha que 'texts' e 'labels' sejam seus dados de treinamento e teste
# Certifique-se de que 'labels' contenha categorias de 1, 0 e -1 codificadas como 0, 1 e 2.

# Carregando o CSV
df = pd.read_csv('Twitter_Data.csv')  # Substitua 'seuarquivo.csv' pelo nome do seu arquivo CSV

# 1. Pré-processamento de Dados
texts = df['clean_text'].tolist()
labels = df['category'].tolist()


# Mapeando rótulos para 0 (negativo), 1 (neutro), 2 (positivo)
label_mapping = {-1: 0, 0: 1, 1: 2}

labels = [label_mapping[label] if label in label_mapping else -1 for label in labels]
# Hiperparâmetros
max_words = 10000
max_sequence_length = 144
embedding_dim = 144  # Dimensão das incorporações do Word2Vec


# Preprocess the text data
special_chars = string.punctuation
stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = str(text).lower()
    text = demoji.replace(text, '')
    text = re.sub(f"[{special_chars}]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


texts = [preprocess(text) for text in texts]


# Treine o modelo Word2Vec
word2vec_model = Word2Vec(texts, vector_size=embedding_dim, window=5, min_count=1, sg=0)

# Pré-processamento dos dados
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

labels = tf.keras.utils.to_categorical(labels, num_classes=3)

# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Crie a matriz de incorporação de palavras usando o Word2Vec
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words and word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

# Crie o modelo RNN com as incorporações do Word2Vec
model = Sequential(name="LSTM_Model")
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length, weights=[embedding_matrix], trainable=False))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.add(Dense(3, activation='sigmoid'))

# Compile o modelo
model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])

# Treine o modelo
model.fit(X_train, y_train, epochs=10,  batch_size=64, validation_data=(X_test, y_test))

# Avalie o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

# Faça previsões com o modelo
predictions = model.predict(X_test)
