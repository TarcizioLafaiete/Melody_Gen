import json
import glob

from music21 import converter, instrument, note, chord


notes = []

file = "beethoven_opus10_3.mid"
midi = converter.parse(file)
parse = None

parts = instrument.partitionByInstrument(midi)

# parts.show("text")

if parts:
    parse = parts.parts[1].recurse()
else:
    parse = midi.flat.notes

# print(parse)

for element in parse:
    if isinstance(element,note.Note):
        notes.append((str(element.pitch),str(element.quarterLength)))
    elif isinstance(element,chord.Chord):
        notes.append((('.'.join(str(n) for n in element.normalOrder)),str(element.quarterLength)))
    elif isinstance(element,note.Rest):
        notes.append(("Rest",str(element.quarterLength)))

print(notes)