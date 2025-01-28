import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import json
import numpy as np

import constants
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from model.lstm_model import Melody_LSTM
# from model.temporal_model import TimeSeries_Melody
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

    offset_map = {}
    with open(constants.OFFSET_LABEL,'r') as file:
        data = json.load(file)
    offset_map = data['original']

    return notes_map,offset_map

def main():

    configure_gpu()

    # loader = DatasetLoader()
    # encoder,num_classes = loader.getEncoderFeatures()
    # data = loader.getDataset()

    n_map,d_map = get_maps()

    train_gen = MelodyDataGenerator(constants.TRAIN_FILE,len(n_map))
    # val_gen = MelodyDataGenerator(val,num_classes

    train_notes_seq,train_notes = train_gen.getNotesSeq()
    train_duration_seq, train_duration = train_gen.getDurationSeq()

    train_notes = to_categorical(train_notes,num_classes=len(n_map))
    train_duration = to_categorical(train_duration,num_classes=len(d_map))

    print(train_notes_seq.shape)
    print(train_duration_seq.shape)

    # val_seq,val_notes = val_gen.genSeq()
    # val_notes = to_categorical(val_notes,num_classes=num_classes)

    tensor_input = (tf.convert_to_tensor(train_notes_seq,dtype=tf.float32),
                    tf.convert_to_tensor(train_duration_seq,dtype=tf.float32))
    
    tensor_output = (tf.convert_to_tensor(train_notes,dtype=tf.float32),
                     tf.convert_to_tensor(train_duration,dtype=tf.float32))

    melodyModel = Melody_LSTM(constants.SEQUENCE_LEN,len(n_map),len(d_map))
    melodyModel.compile([["accuracy"],["accuracy"]])
    melodyModel.fit(tensor_input,tensor_output)
if __name__ == "__main__":
    main()