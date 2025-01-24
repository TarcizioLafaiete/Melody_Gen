import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import json
import numpy as np

import constants
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
# from model.lstm_model import Melody_LSTM
from model.temporal_model import TimeSeries_Melody
from data_process.data_generator import MelodyDataGenerator
from data_process.dataset_loader import DatasetLoader


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

    return notes_map

def main():

    configure_gpu()

    loader = DatasetLoader()
    encoder,num_classes = loader.getEncoderFeatures()
    train,val,test = loader.getDataset()

    train_gen = MelodyDataGenerator(train,num_classes)
    val_gen = MelodyDataGenerator(val,num_classes)

    sequences,next_notes = train_gen.genSeq()
    next_notes = to_categorical(next_notes,num_classes=num_classes)

    # n_map = get_maps()

    melodyModel = TimeSeries_Melody(constants.SEQUENCE_LEN,num_classes)
    melodyModel.compile(["accuracy"])
    melodyModel.fit(train_gen,val_gen)


if __name__ == "__main__":
    main()