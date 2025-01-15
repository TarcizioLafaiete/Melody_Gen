import sys
import json
import constants

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
import numpy as np


class MelodyDataGenerator(Sequence):
    def __init__(self,json_file,data_size,augment=False):
        
        self.batch_size = constants.BATCH_SIZE
        self.sequence_len = constants.SEQUENCE_LEN
        self.augment = augment
        self.data_size  = data_size
        self.indexes = np.arange(self.data_size)
        self.batch_array = np.arange(self.batch_size)

        self.dataset = {}
        with open(json_file,'r') as file:
            self.dataset = json.load(file)
        
        self.notes_map = {}
        with open(constants.NOTES_LABEL,'r') as file:
            data = json.load(file)
        self.notes_map = data['original']

        self.offset_map = {}
        with open(constants.OFFSET_LABEL,'r') as file:
            data = json.load(file)
        self.offset_map = data['original']

    def __len__(self):
        return int(np.ceil(self.data_size / self.batch_size))
    
    def __getitem__(self, index):
        
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, self.data_size)

        notes_inputNetwork,notes_outputNetwork = self.__generate_Input_and_Output(start,end,'notes',self.notes_map)
        offset_inputNetwork,offset_outputNetwork = self.__generate_Input_and_Output(start,end,'offset',self.offset_map)

        # print("Vetores de entrada e saída gerados")

        notes_inputNetwork = self.__input_normalize(notes_inputNetwork,self.sequence_len,self.notes_map)
        offset_inputNetwork = self.__input_normalize(offset_inputNetwork,self.sequence_len,self.offset_map)

        # print("Entadas Normalizadas")

        notes_outputNetwork = to_categorical(notes_outputNetwork,num_classes=len(self.notes_map))
        offset_outputNetwork = to_categorical(offset_outputNetwork,num_classes=len(self.offset_map))


        return (
        (
            tf.convert_to_tensor(notes_inputNetwork, dtype=tf.float32),
            tf.convert_to_tensor(offset_inputNetwork, dtype=tf.float32),
        ),
        (
            tf.convert_to_tensor(notes_outputNetwork, dtype=tf.float32),
            tf.convert_to_tensor(offset_outputNetwork, dtype=tf.float32),
        ))
        
        # return [notes_inputNetwork,offset_inputNetwork], [notes_outputNetwork,offset_outputNetwork]

    
    def __generate_Input_and_Output(self,start:int,end:int,atribute:str,map:dict):

        inputNet = []
        outputNet = []
        
        limit = end - start
        for i in self.batch_array:
            if i > limit:
                break
            values = self.dataset[f"music_{i + start + constants.MUSIC_MIN_INDEX}"][atribute]
            first_n_value = values[:self.sequence_len]
            others_value = values[self.sequence_len:]
            
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
        return input_seq/len(map)


            
if __name__ == "__main__":
    size = constants.MUSIC_MAX_INDEX - constants.MUSIC_MIN_INDEX
    train_gen = MelodyDataGenerator(constants.TRAIN_FILE,int(np.ceil(size * constants.TRAIN_PERCENTAGE)))
    val_gen = MelodyDataGenerator(constants.VALIDATION_FILE,int(np.ceil(size *(1 - constants.TRAIN_PERCENTAGE))))

    print(train_gen.__len__())
    print(val_gen.__len__())