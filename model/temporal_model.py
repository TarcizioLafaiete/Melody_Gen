import numpy as np
import constants

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Concatenate, Embedding
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential


class TimeSeries_Melody:
    def __init__(self,input_shape,note_output):
     # Definir as entradas
        self.model = Sequential([
        Embedding(input_dim=note_output, output_dim=100, input_length=constants.SEQUENCE_LEN),
        LSTM(256, return_sequences=True),
        LSTM(256),
        Dense(note_output, activation='softmax')
        ])

        # Resumo do modelo
        self.model.summary()
    
    def compile(self,metrics):
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=constants.LOSS,metrics=metrics)

    def getModel(self):
        return self.model

    def fit(self,train_gen,val_gen):
        filepath = "weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.keras"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]

        self.model.fit(train_gen,validation_data=val_gen,epochs=constants.EPOCHS,batch_size=constants.BATCH_SIZE, callbacks=callbacks_list)


if __name__ == "__main__":
    m = TimeSeries_Melody(constants.SEQUENCE_LEN,30)