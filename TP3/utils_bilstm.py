# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:41:10 2021
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from keras import backend as K
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, Dropout

def to_categorical(sequences, categories):
    # A partir d'une séquences de tags et d'un nombre de catégories, conversion des tags en vecteur one-hot
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
        
    return np.array(cat_sequences)


def ignore_class_accuracy(to_ignore=0):
    #Calcul de l'accuracy en ignorant la classe de padding
    
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_true_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    
    return ignore_accuracy


def categorical_crossentropy_ignore():
    #Fonction de loss en ignorant les faux négatifs du padding (mais pas les vrais positifs)
    def loss(y_true, y_pred):
        
        y_true_class = K.argmax(y_true, axis=-1)

        # Trouve les indices de padding
        indices = tf.where(tf.math.equal(y_true_class, 0))
        
        # Créer un tenseur avec le nombre de fois où on doit remplacer le padding
        updates = tf.repeat(
            [[1.0] + [0.0 for i in range(y_true.shape[2]-1)]],
            repeats=indices.shape[0],
            axis=0)
        
        # Remplacement des valeurs du padding 
        yp = tf.tensor_scatter_nd_update(y_pred, indices, updates)

        yp /= K.sum(yp, axis=-1, keepdims=True)
        yp = K.clip(yp, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(yp)
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


def create_bi_lstm_model(max_length, types_dict, tag_index) : 
    
    model = Sequential()
    model.add(InputLayer(input_shape=(max_length, )))
    model.add(Embedding(len(types_dict), 128))
    #Dropout
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(tag_index))))
    model.add(Activation('softmax'))
    
    # Ajout d'un early stopping au bout de 3 époques sans amélioration de la loss
    early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    
    # Sauvegarde du meilleur modèle pour lequel l'accuracy est au maximum en ignorant la classe de padding
    model_checkpoint_callback = ModelCheckpoint(
    filepath='./history_checkpoint',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
    
    model.compile(loss=categorical_crossentropy_ignore(),
                  optimizer=Adam(0.001),
                  metrics=["accuracy", ignore_class_accuracy(0)],
                  run_eagerly=True)
    
    model.summary()
    
    return model, early_stopping_callback, model_checkpoint_callback


def generate_model_history(history, monitor, ylabel, title):
    plt.plot(history.history[monitor])
    plt.plot(history.history['val_' + monitor])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Epochs')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    
    
def logits_to_tokens(sequences, index):
    #Choisi l'argument maximum de chaque prédiction
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences


def transform_output(output_bi_lstm, lens_sentences) :
    clean_output = []
    
    for i in range(len(lens_sentences)) :
        clean_output += list(output_bi_lstm[i][0:lens_sentences[i]])
    
    return clean_output


def get_output_bilstm(model, pad_sentences, reverse_tag_index, lens_sentences) :
    # Prédictions des classes à partir du modèle donné
    predictions_bi_lstm = model.predict(pad_sentences)
    # Calcul de la classe la plus probable pour chaque mot
    output_bi_lstm = logits_to_tokens(predictions_bi_lstm, reverse_tag_index)
    # Nettoyage de l'output en enlevant le padding
    clean_output_bi_lstm = transform_output(output_bi_lstm, lens_sentences)
    
    return clean_output_bi_lstm 