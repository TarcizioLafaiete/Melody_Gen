import json
import glob

from music21 import converter, instrument, note, chord

def generate_notesArray():

    notes = []
    duration = []

    for file in glob.glob("midi_songs/*.midi"):
        midi = converter.parse(file)
        parse = None

        parts = instrument.partitionByInstrument(midi)

        if parts:
            parse = parts.parts[0].recurse()
        else:
            parse = midi.flat.notes

        last_offset = 0
        symb = ""

        for element in parse:

            time_diff = element.offset - last_offset
            last_offset = element.offset
            duration.append(float(time_diff))

            if isinstance(element,note.Note):
                symb = str(element.pitch)
            elif isinstance(element,chord.Chord):
                symb = '.'.join(str(n) for n in element.normalOrder)
            elif isinstance(element,note.Rest):
                symb = "Rest"
            notes.append((symb,str(element.quarterLength))) 
    print(f"Tamanho do array: {len(notes)} - variedade: {len(set(notes))}")
    print(f"Tamanho do array: {len(duration)} - variedade: {len(set(duration))}")

    data = {"notes": notes, "durations": duration}
    with open("notes_durations.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


generate_notesArray()