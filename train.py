import json
import numpy as np

import constants
import tensorflow as tf
from model.lstm_model import Melody_LSTM
from data_process.data_generator import MelodyDataGenerator

def get_maps():
    notes_map = {}
    with open(constants.NOTES_LABEL,'r') as file:
        data = json.load(file)
    notes_map = data['original']

    offset_map = {}
    with open(constants.OFFSET_LABEL,'r') as file:
        data = json.load(file)
    offset_map = data['original']

    return notes_map,offset_map

def define_output_signature(n_map,o_map):
    notes_input_shape = (None, constants.SEQUENCE_LEN, len(n_map))
    offset_input_shape = (None, constants.SEQUENCE_LEN, len(o_map))

    notes_output_shape = (None, len(n_map))
    offset_output_shape = (None, len(o_map))

    output_signature = (
        [
            tf.TensorSpec(shape=notes_input_shape, dtype=tf.float32),  # notes_inputNetwork
            tf.TensorSpec(shape=offset_input_shape, dtype=tf.float32), # offset_inputNetwork
        ],
        [
            tf.TensorSpec(shape=notes_output_shape, dtype=tf.float32), # notes_outputNetwork
            tf.TensorSpec(shape=offset_output_shape, dtype=tf.float32), # offset_outputNetwork
        ]
    )
    return output_signature

def main():
    size = constants.MUSIC_MAX_INDEX - constants.MUSIC_MIN_INDEX
    train_gen = MelodyDataGenerator(constants.TRAIN_FILE,int(np.ceil(size * constants.TRAIN_PERCENTAGE)))
    val_gen = MelodyDataGenerator(constants.VALIDATION_FILE,int(np.ceil(size * (1 - constants.TRAIN_PERCENTAGE))))

    n_map,o_map = get_maps()

    # out_sig = define_output_signature(n_map,o_map)

    # train_dataset = tf.data.Dataset.from_generator(
    # generator=lambda: (train_gen[i] for i in range(len(train_gen))),  # O gerador da classe que você implementou
    # output_signature=out_sig
    # )

    # val_dataset = tf.data.Dataset.from_generator(
    # generator=lambda: (val_gen[i] for i in range(len(val_gen))),  # O gerador da classe que você implementou
    # output_signature=out_sig
    # )


    melodyModel = Melody_LSTM(constants.SEQUENCE_LEN,len(n_map),len(o_map))
    melodyModel.compile([["accuracy"], ["accuracy"]])
    model = melodyModel.getModel()

    for inputs,labels in train_gen:
        
        print(len(labels[1][0]))
        model.fit(inputs, labels, epochs=1, batch_size=constants.BATCH_SIZE, verbose=1)



if __name__ == "__main__":
    main()