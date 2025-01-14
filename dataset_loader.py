import sys
import json

from tensorflow.keras.utils import to_categorical
import numpy as np


class DatasetLoader:
    def __init__(self,json_file,sequence_length,n_limit):
        complete_data = {}

        self.notes_inputNetwork = []
        self.notes_outputNetwork = []

        self.offset_inputNetwork = []
        self.offset_outputNetwork = []
        
        with open(json_file,'r') as file:
            complete_data = json.load(file)

        dataset_data = {key: value for key, value in complete_data.items() 
                if key.startswith("music_") and int(key.split("_")[1]) <= n_limit}

        for key in dataset_data:
            dataset_data[key]['notes'] = [f"{v[0]};{v[1]}" for v in dataset_data[key]['notes']]
        
        self.notes_map = self.__get_data_unique_mapping(dataset_data,"notes")
        self.offset_map = self.__get_data_unique_mapping(dataset_data,"offset")

        print("Dados Mapeados")


        self.notes_inputNetwork,self.notes_outputNetwork = self.__generate_Input_and_Output(dataset_data,"notes",sequence_length,self.notes_map)
        self.offset_inputNetwork,self.offset_outputNetwork = self.__generate_Input_and_Output(dataset_data,"offset",sequence_length,self.offset_map)

        print("Vetores de entrada e saída gerados")

        self.notes_inputNetwork = self.__input_normalize(self.notes_inputNetwork,sequence_length,self.notes_map)
        self.offset_inputNetwork = self.__input_normalize(self.offset_inputNetwork,sequence_length,self.offset_map)

        print("Entadas Normalizadas")

        self.notes_outputNetwork = to_categorical(self.notes_outputNetwork)
        self.offset_outputNetwork = to_categorical(self.offset_outputNetwork)

        print("Saida Normalizada")


    def get_Input_sequencer(self):
        return self.notes_inputNetwork,self.offset_inputNetwork
    
    def get_Output_sequencer(self):
        return self.notes_outputNetwork,self.offset_outputNetwork
    
    def __get_data_unique_mapping(self,dataset_data,atribute):
        unique_set = set()

        for music_key, music_data in dataset_data.items():
            values = music_data.get(atribute, [])
            unique_set.update(values)  

        return {value: int(idx) for idx, value in enumerate(sorted(unique_set))}
    
    def __generate_Input_and_Output(self,dataset_data: dict,atribute:str,sequence_length: int,map:dict):

        inputNet = []
        outputNet = []
        for music_key, music_data in dataset_data.items():
            values = music_data.get(atribute,[])
            first_n_value = values[:sequence_length]
            others_value = values[sequence_length:]
            
            for value in others_value:
                inputNet.append([map[char] for char in first_n_value])
                outputNet.append(map[value])
                first_n_value.pop(0)
                first_n_value.append(value)
        return inputNet,outputNet
    
    def __input_normalize(self,input_seq,sequence_length,map):
        
        n_patterns = len(input_seq)
        input_seq = np.reshape(input_seq,(n_patterns,sequence_length,1))
        return input_seq/len(map)


            
if __name__ == "__main__":
    loader = DatasetLoader(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
    ni,oi = loader.get_Input_sequencer()
    no,oo = loader.get_Output_sequencer()

    print(ni.shape)
    print(no.shape)

    print('\n')

    print(oi.shape)
    print(oo.shape)