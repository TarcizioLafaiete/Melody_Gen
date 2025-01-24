#Arquivo json utilizando atualmente
JSON_FILE = "files/classical_piano.json"
TRAIN_FILE = "files/train.json"
VALIDATION_FILE = "files/validation.json"
TEST_FILE = "files/test.json"

#Constantes relacionadas com o tamanho do conjunto de dados utilizados no treinamento
MUSIC_MIN_INDEX = 0
MUSIC_MAX_INDEX = 58
TRAIN_PERCENTAGE = 0.8

#Arquivos de maps e reversos
NOTES_LABEL = "files/notes_labels.json"

#Constantes de treinamento e do modelo
SEQUENCE_LEN = 50
BATCH_SIZE = 64
EPOCHS = 200
LOSS = "categorical_crossentropy"
OPTIMIZER = "adam"
NET_QUOCIENT=1
DROPOUT = 0.2
MUSIC_LEN = 500
MUSIC_TIME_STEP = 0.25
