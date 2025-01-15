import sys
import json
import constants

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
import numpy as np


class DatasetLoader:
    def __init__(self):
        complete_data = {}

        json_file = constants.JSON_FILE
        self.min = constants.MUSIC_MIN_INDEX
        self.max = constants.MUSIC_MAX_INDEX
        
        with open(json_file,'r') as file:
            complete_data = json.load(file)

        dataset_data = self.__select_data(complete_data)

        train_data,val_data = self.__split_data(dataset_data)
        with open(constants.TRAIN_FILE,'w') as file:
            json.dump(train_data,file,indent=1)
        with open(constants.VALIDATION_FILE,'w') as file:
            json.dump(val_data,file,indent=1)
        print("Dados cortados e salvos")

        # dataset_data = self.__concat_notes_and_duration(dataset_data)
        # print("Tratamento dos dados")

        self.notes_map = self.__get_data_unique_mapping(dataset_data,"notes")
        self.offset_map = self.__get_data_unique_mapping(dataset_data,"offset")
        print("Dados Mapeados")

        self.notes_reverse = self.__generate_reverse_map(self.notes_map)
        self.offset_reverse  = self.__generate_reverse_map(self.offset_map)
        print("Mapas reversos criados")

        self.__save_map_and_reverse(self.notes_map,self.notes_reverse,constants.NOTES_LABEL)
        self.__save_map_and_reverse(self.offset_map,self.offset_reverse,constants.OFFSET_LABEL)
        print("Mapas e reversos salvos")
 

    def get_maps(self):
        return self.notes_map,self.offset_map
    
    def __save_map_and_reverse(self,map,reverse,file):
        data = {
            "original": map,
            "reverse": reverse
        }

        with open(file,"w") as f:
            json.dump(data,f,indent=1)
    
    def __select_data(self,data):
        return {key: value for key, value in data.items() 
                if key.startswith("music_") and int(key.split("_")[1]) < self.max and int(key.split("_")[1]) >= self.min}

    def __split_data(self,data):
        split_point = int(np.ceil((self.max - self.min)*constants.TRAIN_PERCENTAGE)) +  self.min
        train_data = {}
        val_data = {}
        for key,value in data.items():
            if key.startswith("music_") and int(key.split("_")[1]) < split_point:
                train_data[key] = value
            else:
                val_data[key] = value 
        return train_data,val_data               

    def __get_data_unique_mapping(self,dataset_data,atribute):
        unique_set = set()

        for music_key, music_data in dataset_data.items():
            values = music_data.get(atribute, [])
            unique_set.update(values)  

        return {value: int(idx) for idx, value in enumerate(sorted(unique_set))}
    
    def __generate_reverse_map(self,map):
        reverse = {v: k for k, v in map.items()}    
        return reverse


            
if __name__ == "__main__":
    loader = DatasetLoader()
