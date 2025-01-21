import sys
import json
import glob

import music21 as m21
from music21 import converter, instrument, note, chord

def generate_notesArray():
    music_num = 0
    data = {}

    for file in glob.glob("midi_songs/*.midi"):
        midi = converter.parse(file)
        parse = None
        notes = []

        # Obter a tonalidade da música
        key = midi.analyze('key')

        parts = instrument.partitionByInstrument(midi)

        if parts:
            parse = parts.parts[0].recurse()
        else:
            parse = midi.flat.notes

        symb = ""

        for element in parse:
            if isinstance(element, note.Note):
                symb = str(element.pitch)
            elif isinstance(element, chord.Chord):
                symb = '.'.join(str(n) for n in element.pitches)
            elif isinstance(element, note.Rest):
                symb = "Rest"
            notes.append((symb, f"{element.quarterLength}"))

        # Salvar dados da música
        data[f"music_{music_num}"] = {
            "notes": notes,
            "key": {
                "tonic": str(key.tonic),          # Tônica da tonalidade
                "mode": str(key.mode),           # Modo (maior, menor, etc.)
                "name": str(key)                 # Nome completo (ex.: "C major")
            }
        }

        # Escrever os dados no arquivo JSON
        with open(sys.argv[1], "w+") as json_file:
            json.dump(data, json_file, indent=2)

        music_num += 1

generate_notesArray()
