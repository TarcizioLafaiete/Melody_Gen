import numpy as np
import constants

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Concatenate
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint


class Melody_LSTM:
    def __init__(self,input_shape,note_output,offset_output):
     # Definir as entradas
        inputNote = Input(shape=(input_shape, 1))  # Exemplo com 10 recursos por timestep
        inputOff = Input(shape=(input_shape, 1))  # Outra entrada com 10 recursos por timestep

        # Primeiro caminho para a primeira entrada
        xNote = LSTM(128, return_sequences=True)(inputNote)
        xNote = Dropout(0.5)(xNote)

        # Segundo caminho para a segunda entrada
        xOff = LSTM(128, return_sequences=True)(inputOff)
        xOff = Dropout(0.5)(xOff)

        # Concatenar as duas saídas das LSTM
        merged = Concatenate()([xNote, xOff])

        # Adicionar LSTM, Dropout e Dense após a concatenação
        x = LSTM(256, return_sequences=True)(merged)
        x = Dropout(0.3)(x)
        x = LSTM(256)(x)
        x = BatchNorm()(x)
        x = Dropout(0.3)(x)
        x = Dense(128,activation='relu')(x)

        # Camadas Dense para a primeira saída
        outputNote = Dense(64, activation='relu')(x)
        x = BatchNorm()(x)
        outputNote = Dropout(0.3)(outputNote)
        outputNote = Dense(note_output, activation='softmax')(outputNote)  # Saída binária, por exemplo

        # Camadas Dense para a segunda saída
        outputOff = Dense(64, activation='relu')(x)
        x = BatchNorm()(x)
        outputOff = Dropout(0.3)(outputOff)
        outputOff = Dense(offset_output, activation='softmax')(outputOff)  # Saída binária, por exemplo

        # Criar o modelo com as duas entradas e saídas
        self.model = Model(inputs=[inputNote, inputOff], outputs=[outputNote, outputOff])

        # Resumo do modelo
        self.model.summary()
    
    def compile(self,metrics):
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=constants.LOSS,metrics=metrics)

    def getModel(self):
        return self.model

    def fit(self,train_gen,val_gen,train_len,val_len):
        filepath = "weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.keras"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]

        self.model.fit(train_gen,validation_data=val_gen,steps_per_epoch=train_len,validation_steps=val_len,epochs=constants.EPOCHS, callbacks=callbacks_list)


if __name__ == "__main__":
    m = Melody_LSTM(constants.SEQUENCE_LEN,30,4)