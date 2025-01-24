import sys
import json
import constants

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
import numpy as np

from data_process.dataset_loader import DatasetLoader

class MelodyDataGenerator(Sequence):
    def __init__(self,data,num_classes,augment=False, **kwargs):

        super().__init__(**kwargs)
        
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
        batch_indexes = self.indexes[start:end]
        print(batch_indexes)

        notes_inputNetwork = np.array([self.noteIn[i] for i in batch_indexes])
        notes_outputNetwork = np.array([self.noteOut[i] for i in batch_indexes])

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
    loader = DatasetLoader()
    t,_,_ = loader.getDataset()
    _,num_classes = loader.getEncoderFeatures()
    train_gen = MelodyDataGenerator(t,num_classes)
    for i in range(len(train_gen)):
        x, y = train_gen[i]
        print(f"Lote {i}: Entrada: {x.shape}, Saída: {y.shape}")

    