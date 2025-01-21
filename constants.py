#Arquivo json utilizando atualmente
JSON_FILE = "files/base_transposed.json"
TRAIN_FILE = "files/train.json"
VALIDATION_FILE = "files/validation.json"

#Constantes relacionadas com o tamanho do conjunto de dados utilizados no treinamento
MUSIC_MIN_INDEX = 0
MUSIC_MAX_INDEX = 10
TRAIN_PERCENTAGE = 0.8

#Arquivos de maps e reversos
NOTES_LABEL = "files/notes_labels.json"
DURATION_LABEL = "files/duration_labels.json"

#Constantes de treinamento e do modelo
SEQUENCE_LEN = 100
BATCH_SIZE = 2
EPOCHS = 200
LOSS = "categorical_crossentropy"
OPTIMIZER = "adam"
NET_QUOCIENT=4
DROPOUT = 0.3
