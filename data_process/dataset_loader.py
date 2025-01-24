
import json
import constants

import numpy as np
from sklearn.preprocessing import LabelEncoder

class DatasetLoader:
    def __init__(self):
        
        with open(constants.JSON_FILE,'r') as f:
            data = json.load(f)

        notes = []
        for key,value in data.items():
            notes.extend([note[0] for note in value])

        self.encoder = LabelEncoder()
        self.encoded_notes = self.encoder.fit_transform(notes)
        self.num_classes = len(self.encoder.classes_)

        self.__split_data()


    def getEncoderFeatures(self):
        return self.encoder,self.num_classes
    
    def getDataset(self):
        return self.train_encoder,self.val_encoder,self.test_encoder
    
    def __split_data(self):
        train_index = int(len(self.encoded_notes)*constants.TRAIN_PERCENTAGE)
        self.train_encoder = self.encoded_notes[:train_index]
        self.val_encoder = self.encoded_notes[train_index:constants.SEQUENCE_LEN*(-1)]
        self.test_encoder = self.encoded_notes[constants.SEQUENCE_LEN*(-1):]
            
if __name__ == "__main__":
    loader = DatasetLoader()
    encoder,num_classes = loader.getEncoderFeatures()
    x_train,x_val,x_test = loader.getDataset()

    print(f"Numero de classes:{num_classes}")
    print(x_test)
