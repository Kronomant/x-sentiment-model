import numpy as np
import gensim
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
nltk.download('punkt')

# Carregando o CSV
df = pd.read_csv('Twitter_Data.csv')  # Substitua 'seuarquivo.csv' pelo nome do seu arquivo CSV

# 1. Pré-processamento de Dados
texts = df['clean_text'].tolist()
labels = df['category'].tolist()

# Mapeando rótulos para 0 (negativo), 1 (neutro), 2 (positivo)
label_mapping = {-1: 0, 0: 1, 1: 2}
labels = [label_mapping[label] for label in labels]

# 2. Word2Vec
sentences = [nltk.word_tokenize(text.lower()) for text in texts]
w2v_model = gensim.models.Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

# 3. Tokenização
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# 4. Padding
max_sequence_length = max(len(sequence) for sequence in sequences)
X = pad_sequences(sequences, maxlen=max_sequence_length)

# 5. Divisão de Dados (simulada, substitua por seus próprios dados)
# Suponha que você queira dividir os dados em treinamento e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 6. Criar a RNN
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sequence_length))
model.add(LSTM(128))
model.add(Dense(3, activation='softmax'))  # Usando 3 unidades de saída para a classificação em três classes
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. Treinamento do Modelo
model.fit(X_train, np.array(y_train), epochs=5, batch_size=64)

# 8. Avaliação do Modelo
loss, accuracy = model.evaluate(X_test, np.array(y_test))
print(f"Loss: {loss}, Accuracy: {accuracy}")

# 9. Previsões (simuladas, substitua por seus próprios dados)
X_new_data = X_test
predictions = model.predict(X_new_data)
print(predictions)
