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

    # loader = DatasetLoader()
    # encoder,num_classes = loader.getEncoderFeatures()
    # data = loader.getDataset()

    n_map = get_maps()

    train_gen = MelodyDataGenerator(constants.TRAIN_FILE,len(n_map))
    # val_gen = MelodyDataGenerator(val,num_classes

    train_seq,train_notes = train_gen.getSeq()
    train_notes = to_categorical(train_notes,num_classes=len(n_map))

    print(train_seq.shape)

    # val_seq,val_notes = val_gen.genSeq()
    # val_notes = to_categorical(val_notes,num_classes=num_classes)

    melodyModel = TimeSeries_Melody(constants.SEQUENCE_LEN,len(n_map))
    melodyModel.compile(["accuracy"])
    melodyModel.fit(train_seq,train_notes)
if __name__ == "__main__":
    main()