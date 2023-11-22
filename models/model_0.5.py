from tensorflow.keras.layers import LSTM,  Embedding, Dense,Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import string
import demoji
import re
from nltk.tokenize import word_tokenize
import os
import json
from nltk.stem import PorterStemmer

ps = PorterStemmer()
max_vocab = 15000
max_len = 500


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

def preprocess_data(data_file):

    messages = []
    labels = []
    if os.path.exists('messages.json') and os.path.exists('labels.json'):
        # Se os arquivos JSON já existirem, carregue-os
        with open('messages.json', 'r') as messages_file, open('labels.json', 'r') as labels_file:
            messages = json.load(messages_file)
            labels = json.load(labels_file)
            
    else:
        data = pd.read_csv(data_file)
        print('Pre-processamento')
        messages = []
        labels = []

        for index, row in data.iterrows():
            processed_message = process_message(row['clean_text'])  # Processa a mensagem
            messages.append(processed_message)
            if row['category'] == -1:
                labels.append(0)
            elif row['category'] == 0:
                labels.append(1)
            else:
                labels.append(2)
    # Salvar os objetos em arquivos JSON
        with open('messages.json', 'w') as messages_file, open('labels.json', 'w') as labels_file:
            json.dump(messages, messages_file)
            json.dump(labels, labels_file)


    messages = np.asarray(messages)
    labels_array = np.asarray(labels)

    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(messages)
    sequences = tokenizer.texts_to_sequences(messages)

    word_index = tokenizer.word_index
    word_index['<UNK>'] = len(word_index) + 1



    return sequences, labels_array, word_index, max_len

def main(rnn_model, data_file):
    sequences, labels, word_index, max_len = preprocess_data(data_file)
    print("Number of messages: ", len(sequences))
    print("Number of labels: ", len(labels))

    data = pad_sequences(sequences, maxlen=max_len)
    print("data shape: ", data.shape)

    # Normalização dos dados
    data = pad_sequences(sequences, maxlen=max_len)
    data = (data - data.mean()) / data.std()

    # Divisão dos dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    embedding_dim = 100  # Dimensão do embedding (ajuste conforme necessário)
    hidden_units = 128  # Número de unidades LSTM

    model = Sequential()
    model.add(Embedding(input_dim=len(word_index), output_dim=embedding_dim, input_length=max_len))
    model.add(LSTM(hidden_units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(hidden_units))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    # Compilação do modelo
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Treinamento do modelo
    batch_size = 64
    epochs = 10
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)
    model.summary()

    model.save(rnn_model + '.keras')
   # model.fit(X_test, labels_train, epochs=12, batch_size=60, validation_split=0.2)
   # acc = model.evaluate(messages_test, labels_test)
   # print("Test loss is {0:.2f} accuracy is {1:.2f}".format(acc[0], acc[1]))

def load_and_classify(model_path, custom_msg, word_index, max_len):
    loaded_model = load_model(model_path)
    def message_to_array(msg):
        msg = msg.lower().split(' ')
        print(msg)
        test_seq = [word_index[word] if word in word_index and word_index[word] < max_vocab else word_index['<UNK>'] for word in msg]
        test_seq = [idx if idx < max_vocab else max_vocab - 1 for idx in test_seq]

        test_seq = np.pad(test_seq, (max_len - len(test_seq), 0), 'constant', constant_values=(0))
        test_seq = test_seq.reshape(1, max_len)
        return test_seq

    test_seq = message_to_array(custom_msg)
    pred = loaded_model.predict(test_seq)
    class_names = ["negative", "neutral", "positive"]
    print(pred)
    pred = loaded_model.predict(test_seq)


    # Defina o limiar de decisão
    threshold = 0.3

    # Obtenha o índice da classe com maior probabilidade
    predicted_class_index = (pred > threshold).argmax()

    # Use o índice para obter a classe correspondente
    predicted_class = class_names[predicted_class_index]

    print(f"A classificação prevista é: {predicted_class}")

    #for i in range(len(class_names)):
        #print(f"{class_names[i]}: {pred[0][i]:.2f}")

if __name__ == '__main__':
    data_file = "Twitter_Data.csv"
    main('SimpleRNN', data_file)
    print('Teste')
    model_path = 'SimpleRNN.keras'  # Substitua pelo caminho correto para o arquivo .keras
    custom_msg = 'but arrogant modi sarkar destroyed msmes for yrs with the worlds most complicated with rates 1000 modifications'  # Sua mensagem de exemplo
    #_ , _, word_index, max_len = preprocess_data(data_file)
    # Chame a função para carregar o modelo e fazer a classificação
    #load_and_classify(model_path, custom_msg, word_index, max_len)
