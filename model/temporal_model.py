import numpy as np
import constants

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Concatenate, Embedding
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential


class TimeSeries_Melody:
    def __init__(self,seq_len,vocab_size):
     # Definir as entradas
        self.model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, input_length=seq_len),
        LSTM(512, return_sequences=True),
        Dropout(constants.DROPOUT),
        LSTM(512),
        BatchNorm(),
        Dense(256,activation='relu'),
        Dropout(constants.DROPOUT),
        Dense(vocab_size, activation='softmax')
        ])

        # Resumo do modelo
        self.model.summary()
    
    def compile(self,metrics):
        self.model.compile(optimizer=constants.OPTIMIZER, loss=constants.LOSS,metrics=metrics)

    def getModel(self):
        return self.model

    def fit(self,input,output):
        filepath = "weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.keras"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]

        # train_gen.on_epoch_end()

        self.model.fit(input,
                        output,
                        validation_split=0.2,
                        # validation_data =(val_input,val_output),
                       epochs=constants.EPOCHS,
                       batch_size=constants.BATCH_SIZE, 
                       callbacks=callbacks_list)


if __name__ == "__main__":
    seq_len = 10  # Por exemplo
    vocab_size = 50  # Por exemplo
    mock_input = np.random.randint(0, vocab_size, (32, seq_len))
    mock_output = to_categorical(np.random.randint(0, vocab_size, (32,)), num_classes=vocab_size)

    model = TimeSeries_Melody(seq_len, vocab_size)
    model.compile(["accuracy"])
    model.fit(mock_input, mock_output)