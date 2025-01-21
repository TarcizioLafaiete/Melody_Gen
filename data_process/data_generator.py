import sys
import json
import constants

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
import numpy as np


class MelodyDataGenerator(Sequence):
    def __init__(self,json_file,augment=False):
        
        self.batch_size = constants.BATCH_SIZE
        self.sequence_len = constants.SEQUENCE_LEN
        self.augment = augment
        self.batch_array = np.arange(self.batch_size)

        dataset = {}
        with open(json_file,'r') as file:
            dataset = json.load(file)
        
        self.notes_map = {}
        with open(constants.NOTES_LABEL,'r') as file:
            data = json.load(file)
        self.notes_map = data['original']

        self.noteIn,self.noteOut = self.__get_Input_and_Output(dataset,self.notes_map)

        self.data_size = len(self.noteOut)
        self.indexes = np.arange(self.data_size)

    def __len__(self):
        return int(np.ceil(self.data_size / self.batch_size)) 
    
    def __getitem__(self, index):
        
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, self.data_size)

        notes_inputNetwork = self.noteIn[start:end]
        notes_outputNetwork = self.noteOut[start:end]

        # print("Vetores de entrada e saída gerados")

        notes_inputNetwork = self.__input_normalize(notes_inputNetwork,self.sequence_len,self.notes_map)
        # print("Entadas Normalizadas")

        notes_outputNetwork = to_categorical(notes_outputNetwork,num_classes=len(self.notes_map))


        return (
            tf.convert_to_tensor(notes_inputNetwork, dtype=tf.float32),
            tf.convert_to_tensor(notes_outputNetwork, dtype=tf.float32),
        )
        
        # return [notes_inputNetwork,offset_inputNetwork], [notes_outputNetwork,offset_outputNetwork]
    
    def __get_Input_and_Output(self,data:dict,map:dict):
        
        inputNet = []
        outputNet = []
        for music_key, music_data in data.items():
            first_n_value = music_data[:self.sequence_len]
            others_value = music_data[self.sequence_len:]
            
            for value in others_value:
                value = str(value)
                inputNet.append([map[f"{char}"] for char in first_n_value])
                outputNet.append(map[value])
                first_n_value.pop(0)
                first_n_value.append(value)
        return inputNet,outputNet


    def __input_normalize(self,input_seq,sequence_length,map):
        
        n_patterns = len(input_seq)
        input_seq = np.reshape(input_seq,(n_patterns,sequence_length,1))
        return input_seq/float(len(map))


            
if __name__ == "__main__":
    size = constants.MUSIC_MAX_INDEX - constants.MUSIC_MIN_INDEX
    train_gen = MelodyDataGenerator(constants.TRAIN_FILE,int(np.ceil(size * constants.TRAIN_PERCENTAGE)))
    val_gen = MelodyDataGenerator(constants.VALIDATION_FILE,int(np.ceil(size *(1 - constants.TRAIN_PERCENTAGE))))

    print(train_gen.__len__())
    print(val_gen.__len__())