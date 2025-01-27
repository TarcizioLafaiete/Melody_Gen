from music21 import converter
import sys

# Carregar o arquivo MIDI
midi_path = sys.argv[1]
midi_file = converter.parse(midi_path)

# Exibir a representação textual do arquivo MIDI
midi_file.show('text')
