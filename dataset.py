import json
import glob

from music21 import converter, instrument, note, chord

def generate_notesArray():

    notes = []

    for file in glob.glob("midi_songs/*.midi"):
        midi = converter.parse(file)
        parse = None

        parts = instrument.partitionByInstrument(midi)

        if parts:
            parse = parts.parts[0].recurse()
        else:
            parse = midi.flat.notes

        for element in parse:
            if isinstance(element,note.Note):
                notes.append((str(element.pitch),str(element.quarterLength)))
            elif isinstance(element,chord.Chord):
                notes.append((('.'.join(str(n) for n in element.normalOrder)),element.quarterLength))
                
