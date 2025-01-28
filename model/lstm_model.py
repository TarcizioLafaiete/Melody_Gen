import numpy as np
import constants

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, concatenate, Embedding, Lambda
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint


class Melody_LSTM:
    def __init__(self,input_shape,note_output,offset_output):
     # Definir as entradas
        inputNote = Input(shape=(input_shape,))  # Exemplo com 10 recursos por timestep
        embeddedNote = Embedding(input_dim=note_output, output_dim=100, input_length=input_shape)(inputNote)
        xNote = LSTM(int(256//constants.NET_QUOCIENT), return_sequences=True)(embeddedNote)
        xNote = Dropout(constants.DROPOUT)(xNote)
        xNote = LSTM(int(512//constants.NET_QUOCIENT))(xNote)
        xNote = BatchNorm()(xNote)
        xNote = Dense(int(256//constants.NET_QUOCIENT),activation='relu')(xNote)
        outputNote = Dropout(constants.DROPOUT)(xNote)
        outputNote = Dense(note_output, activation='softmax',name="Note_Out")(outputNote)  # Saída binária, por exemplo



        inputOff = Input(shape=(input_shape,))  # Outra entrada com 10 recursos por timestep
        embeddedOff = Embedding(input_dim=offset_output, output_dim=100, input_length=input_shape)(inputOff)
        xOff = LSTM(int(256//constants.NET_QUOCIENT), return_sequences=True)(embeddedOff)
        xOff = Dropout(constants.DROPOUT)(xOff)
        xOff = LSTM(int(512//constants.NET_QUOCIENT))(xOff)
        xOff = BatchNorm()(xOff)
        xOff = Dense(int(256//constants.NET_QUOCIENT),activation='relu')(xOff)
        outputOff = Dropout(constants.DROPOUT)(xOff)
        outputOff = Dense(offset_output, activation='softmax',name="Duration_Out")(outputOff)  # Saída binária, por exemplo

        
        # weight1 = tf.Variable(1.0, trainable=True, name="Weight_Input1")
        # weight2 = tf.Variable(1.0, trainable=True, name="Weight_Input2")

        # xNote_weighted = Lambda(lambda x: x[0] * x[1])([xNote, weight1])
        # xOff_weighted = Lambda(lambda x: x[0] * x[1])([xOff, weight2])

        # # Concatenar as duas saídas das LSTM
        # merged = concatenate([xNote_weighted, xOff_weighted])

        # Adicionar LSTM, Dropout e Dense após a concatenação
        # x = LSTM(int(512//constants.NET_QUOCIENT))(merged)
        # x = BatchNorm()(x)
        # x = Dropout(constants.DROPOUT)(x)
        # x = Dense(int(256//constants.NET_QUOCIENT),activation='relu')(x)

        # Camadas Dense para a primeira saída
        
        
        # Criar o modelo com as duas entradas e saídas
        self.model = Model(inputs=[inputNote, inputOff], outputs=[outputNote, outputOff])

        # Resumo do modelo
        self.model.summary()
    
    def compile(self,metrics):
        self.model.compile(optimizer=constants.OPTIMIZER,
                           loss_weights={"Note_Out": 2.0, "Duration_Out": 0.5}, 
                           loss={"Note_Out": constants.LOSS, "Duration_Out": constants.LOSS},
                           metrics=metrics)

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

        self.model.fit(input,
                       output,
                       validation_split=0.2,
                        # validation_data =(val_input,val_output),
                       epochs=constants.EPOCHS,
                       batch_size=constants.BATCH_SIZE, 
                       callbacks=callbacks_list)


if __name__ == "__main__":
    m = Melody_LSTM(constants.SEQUENCE_LEN,30,4)