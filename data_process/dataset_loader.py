import sys
import json

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
import numpy as np


class DatasetLoader:
    def __init__(self,json_file,n_limit,path):
        complete_data = {}

        self.json_file = json_file
        self.min = n_limit[0]
        self.max = n_limit[1]
        
        with open(json_file,'r') as file:
            complete_data = json.load(file)

        dataset_data = self.__split_data(complete_data)
        print("Dados cortados")

        # dataset_data = self.__concat_notes_and_duration(dataset_data)
        # print("Tratamento dos dados")

        self.notes_map = self.__get_data_unique_mapping(dataset_data,"notes")
        self.offset_map = self.__get_data_unique_mapping(dataset_data,"offset")
        print("Dados Mapeados")

        self.notes_reverse = self.__generate_reverse_map(self.notes_map)
        self.offset_reverse  = self.__generate_reverse_map(self.offset_map)
        print("Mapas reversos criados")

        self.__save_map_and_reverse(self.notes_map,self.notes_reverse,f"{path}/notes_labels.json")
        self.__save_map_and_reverse(self.offset_map,self.offset_reverse,f"{path}/offset_labels.json")
        print("Mapas e reversos salvos")
 

    def get_maps(self):
        return self.notes_map,self.offset_map
    
    def get_limits(self):
        return self.min,self.max
    
    def __save_map_and_reverse(self,map,reverse,file):
        data = {
            "original": map,
            "reverse": reverse
        }

        with open(file,"w") as f:
            json.dump(data,f,indent=1)
    
    def __split_data(self,data):
        return {key: value for key, value in data.items() 
                if key.startswith("music_") and int(key.split("_")[1]) < self.max and int(key.split("_")[1]) >= self.min}

    def __concat_notes_and_duration(self,dataset_data):
        for key in dataset_data:
            dataset_data[key]['notes'] = [f"{v[0]};{v[1]}" for v in dataset_data[key]['notes']]
        return dataset_data
    

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
    loader = DatasetLoader(sys.argv[1],(0,30),sys.argv[2])
    print(loader.get_limits())
