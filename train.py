import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import json
import numpy as np

import constants
import tensorflow as tf
from model.lstm_model import Melody_LSTM
from data_process.data_generator import MelodyDataGenerator

def configure_gpu(gpu_index=0):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_index],'GPU')
            print(f"Usando GPU: {gpus[gpu_index].name}")

            tf.config.experimental.set_memory_growth(gpus[gpu_index],True)
        except RuntimeError as e:
            print("Erro ao configurar GPU: {e}")
    else:
        print("Nenhuma GPU encontrada")


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
    notes_input_shape = (None, constants.SEQUENCE_LEN, 1)
    offset_input_shape = (None, constants.SEQUENCE_LEN, 1)

    notes_output_shape = (None, len(n_map))
    offset_output_shape = (None, len(o_map))

    output_signature = (
        (
            tf.TensorSpec(shape=notes_input_shape, dtype=tf.float32),  # notes_inputNetwork
            tf.TensorSpec(shape=offset_input_shape, dtype=tf.float32), # offset_inputNetwork
        ),
        (
            tf.TensorSpec(shape=notes_output_shape, dtype=tf.float32), # notes_outputNetwork
            tf.TensorSpec(shape=offset_output_shape, dtype=tf.float32), # offset_outputNetwork
        )
    )
    return output_signature

def main():

    configure_gpu()

    train_gen = MelodyDataGenerator(constants.TRAIN_FILE)
    val_gen = MelodyDataGenerator(constants.VALIDATION_FILE)

    n_map,o_map = get_maps()

    out_sig = define_output_signature(n_map,o_map)

    train_dataset = tf.data.Dataset.from_generator(
    generator=lambda: (train_gen[i] for i in range(len(train_gen))),  # O gerador da classe que você implementou
    output_signature=out_sig
    )

    val_dataset = tf.data.Dataset.from_generator(
    generator=lambda: (val_gen[i] for i in range(len(val_gen))),  # O gerador da classe que você implementou
    output_signature=out_sig
    )

    melodyModel = Melody_LSTM(constants.SEQUENCE_LEN,len(n_map),len(o_map))
    melodyModel.compile([["accuracy"], ["accuracy"]])
    melodyModel.fit(train_dataset,val_dataset,len(train_gen),len(val_gen))

if __name__ == "__main__":
    main()