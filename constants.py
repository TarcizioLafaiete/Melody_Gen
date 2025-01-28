#Arquivo json utilizando atualmente
JSON_FILE = "files/base_transposed.json"
TRAIN_FILE = "files/train.json"
TEST_FILE = "files/test.json"

#Constantes relacionadas com o tamanho do conjunto de dados utilizados no treinamento
MUSIC_MIN_INDEX = 0
MUSIC_MAX_INDEX = 59
TRAIN_PERCENTAGE = 0.8

#Arquivos de maps e reversos
NOTES_LABEL = "files/notes_labels.json"
OFFSET_LABEL = "files/duration_labels.json"

#Constantes de treinamento e do modelo
SEQUENCE_LEN = 50
BATCH_SIZE = 32
EPOCHS = 200
LOSS = "categorical_crossentropy"
OPTIMIZER = "adam"
NET_QUOCIENT=1
DROPOUT = 0.3

#Constantes da predicao
MUSIC_LEN = 50