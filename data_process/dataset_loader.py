import sys
import json
import constants

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
import numpy as np
from fractions import Fraction


class DatasetLoader:
    def __init__(self):
        complete_data = {}

        json_file = constants.JSON_FILE
        self.min = constants.MUSIC_MIN_INDEX
        self.max = constants.MUSIC_MAX_INDEX
        
        with open(json_file,'r') as file:
            complete_data = json.load(file)

        dataset_data = self.__select_data(complete_data)

        time_series = self.__generate_time_series(dataset_data)
        print("Sequencia temporal gerada")

        train_data,val_data = self.__split_data(time_series)
        with open(constants.TRAIN_FILE,'w') as file:
            json.dump(train_data,file,indent=1)
        with open(constants.TEST_FILE,'w') as file:
            json.dump(val_data,file,indent=1)
        print("Dados cortados e salvos")

        # dataset_data = self.__concat_notes_and_duration(dataset_data)
        # print("Tratamento dos dados")

        self.notes_map = self.__get_data_unique_mapping(time_series)
        print("Dados Mapeados")

        self.notes_reverse = self.__generate_reverse_map(self.notes_map)
        print("Mapas reversos criados")

        self.__save_map_and_reverse(self.notes_map,self.notes_reverse,constants.NOTES_LABEL)
        print("Mapas e reversos salvos")


    def get_maps(self):
        return self.notes_map
    
    def __save_map_and_reverse(self,map,reverse,file):
        data = {
            "original": map,
            "reverse": reverse
        }

        with open(file,"w") as f:
            json.dump(data,f,indent=1)
    
    def __select_data(self,data):
        return {key: value for key, value in data.items() 
                if key.startswith("music_") and int(key.split("_")[1]) <= self.max and int(key.split("_")[1]) >= self.min}

    def __split_data(self,data):
            test_dataset = {}
            test_music = f"music_{self.max-1}"
            test_dataset["test"] = data[test_music]
            train_dataset = {k:v for k,v in data.items() if k != test_music}
            return train_dataset,test_dataset

    def __get_data_unique_mapping(self,dataset_data):
        unique_set = set()

        for _, music_data in dataset_data.items():
            unique_set.update(music_data)  
        return {value: int(idx) for idx, value in enumerate(sorted(unique_set))}
    
    def __generate_reverse_map(self,map):
        reverse = {v: k for k, v in map.items()}    
        return reverse
    
    def __generate_time_series(self,dataset):

        newData = {}
        for music_key, music_data in dataset.items():
            encoded_song = []
            for note,duration in music_data:
                steps = int(float(Fraction(duration)) / constants.MUSIC_TIME_STEP)
                for step in range(steps):
                    if step == 0:
                        encoded_song.append(note)
                    else:
                        encoded_song.append("_")
            newData[music_key] = encoded_song
        
        return newData




            
if __name__ == "__main__":
    loader = DatasetLoader()
