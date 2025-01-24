import sys
import json
import constants

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
import numpy as np


class MelodyDataGenerator(Sequence):
    def __init__(self,data,num_classes,augment=False):
        
        self.batch_size = constants.BATCH_SIZE
        self.sequence_len = constants.SEQUENCE_LEN
        self.augment = augment
        self.batch_array = np.arange(self.batch_size)

        self.dataset = data
        self.num_classes = num_classes
        self.noteIn,self.noteOut = self.__generateSequence()

        self.data_size = len(self.noteOut)
        self.indexes = np.arange(self.data_size)

    def __len__(self):
        return int(np.ceil(self.data_size / self.batch_size)) 
    
    def __getitem__(self, index):
        
        start = index * self.batch_size
        end = min((index + 1)*self.batch_size,self.data_size)

        notes_inputNetwork = self.noteIn[start:end]
        notes_outputNetwork = self.noteOut[start:end]

        notes_outputNetwork = to_categorical(notes_outputNetwork,num_classes=self.num_classes)

        return notes_inputNetwork,notes_outputNetwork
        # return [notes_inputNetwork,offset_inputNetwork], [notes_outputNetwork,offset_outputNetwork]
    
    def on_epoch_end(self):
        """Embaralha os índices após cada época."""
        np.random.shuffle(self.indexes)
    
    def genSeq(self):
        return self.__generateSequence()

    def __generateSequence(self):
        sequences = []
        next_notes = []
        for i in range(len(self.dataset) - constants.SEQUENCE_LEN):
            sequences.append(self.dataset[i:i+constants.SEQUENCE_LEN])
            next_notes.append(self.dataset[i+constants.SEQUENCE_LEN])

        return np.array(sequences), np.array(next_notes)
             
if __name__ == "__main__":
    size = constants.MUSIC_MAX_INDEX - constants.MUSIC_MIN_INDEX
    train_gen = MelodyDataGenerator(constants.TRAIN_FILE,int(np.ceil(size * constants.TRAIN_PERCENTAGE)))
    val_gen = MelodyDataGenerator(constants.VALIDATION_FILE,int(np.ceil(size *(1 - constants.TRAIN_PERCENTAGE))))

    print(train_gen.__len__())
    print(val_gen.__len__())