from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
import pandas as pd

def prepare_tokenizer(texts, max_words=2000):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def prepare_sequences(tokenizer, texts, max_len=150):
    X = tokenizer.texts_to_sequences(texts)
    X = sequence.pad_sequences(X, maxlen=max_len)
    return X

def prepare_labels(labels):
    Y = pd.get_dummies(labels).values
    return Y

def build_model(max_words=2000, embedding_dim=200, num_classes=5):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim))
    model.add(SpatialDropout1D(0.5))
    model.add(GRU(100, dropout=0.5, recurrent_dropout=0.3, return_sequences=True))
    model.add(GRU(50, dropout=0.5, recurrent_dropout=0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['accuracy'])
    return model
