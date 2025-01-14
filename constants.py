#Arquivo json utilizando atualmente
JSON_FILE = "files/notes_and_offsets_v4.json"
SELECTED_JSON_FILE = "files/selected_musics.json"

#Constantes relacionadas com o tamanho do conjunto de dados utilizados no treinamento
MUSIC_MIN_INDEX = 0
MUSIC_MAX_INDEX = 200
TRAIN_PERCENTAGE = 0.8

#Arquivos de maps e reversos
NOTES_LABEL = "files/notes_labels.json"
OFFSET_LABEL = "files/offset_labels.json"

#Constantes de treinamento e do modelo
SEQUENCE_LEN = 100
BATCH_SIZE = 32
EPOCHS = 200
LOSS = ""
OPTIMIZER = ""
