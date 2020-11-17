#!/usr/bin/env python
# coding: utf-8

# In[6]:


# importing libraries
import pandas as pd
import numpy as np
# deep learning library
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# reading csv files
train = pd.read_csv('Train.csv')
valid = pd.read_csv('Valid.csv')

# train_test split
x_tr, y_tr = train['text'].values, train['label'].values

# Tokenize the sentences
tokenizer = Tokenizer()

# preparing vocabulary
tokenizer.fit_on_texts(list(x_tr))

# converting text into integer sequences
x_tr_seq = tokenizer.texts_to_sequences(x_tr)
x_val_seq = tokenizer.texts_to_sequences(x_val)

# padding to prepare sequences of same length
x_tr_seq = pad_sequences(x_tr_seq, maxlen=100)
x_val_seq = pad_sequences(x_val_seq, maxlen=100)

size_of_vocabulary = len(tokenizer.word_index) + 1  # +1 for padding
print(size_of_vocabulary)


model = Sequential()

# embedding layer
model.add(Embedding(size_of_vocabulary, 300, input_length=100, trainable=True))

# lstm layer
model.add(LSTM(128, return_sequences=True, dropout=0.2))

# Global Maxpooling
model.add(GlobalMaxPooling1D())

# Dense Layer
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Add loss function, metrics, optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["acc"])

# Adding callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc',
                     mode='max', save_best_only=True, verbose=1)

# Print summary of model
print(model.summary())


# In[7]:


history = model.fit(np.array(x_tr_seq), np.array(y_tr), batch_size=128, epochs=10, validation_data=(
    np.array(x_val_seq), np.array(y_val)), verbose=1, callbacks=[es, mc])
# loading best model
from keras.models import load_model
model = load_model('best_model.h5')

# evaluation
_, val_acc = model.evaluate(x_val_seq, y_val, batch_size=128)
print(val_acc)


# In[9]:


# load the whole embedding into memory
embeddings_index = dict()
f = open('./glove.6B.300d.txt')

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((size_of_vocabulary, 300))

for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[10]:


model = Sequential()

# embedding layer
model.add(Embedding(size_of_vocabulary, 300, weights=[
          embedding_matrix], input_length=100, trainable=False))

# lstm layer
model.add(LSTM(128, return_sequences=True, dropout=0.2))

# Global Maxpooling
model.add(GlobalMaxPooling1D())

# Dense Layer
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Add loss function, metrics, optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["acc"])

# Adding callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc',
                     mode='max', save_best_only=True, verbose=1)

# Print summary of model
print(model.summary())


# loading best model
from keras.models import load_model
model = load_model('best_model.h5')

# evaluation
_, val_acc = model.evaluate(x_val_seq, y_val, batch_size=128)
print(val_acc)


# In[ ]:


history = model.fit(np.array(x_tr_seq), np.array(y_tr), batch_size=128, epochs=10, validation_data=(
    np.array(x_val_seq), np.array(y_val)), verbose=1, callbacks=[es, mc])
