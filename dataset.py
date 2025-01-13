import json
import glob

from music21 import converter, instrument, note, chord

def generate_notesArray():

    music_num = 0

    for file in glob.glob("midi_songs/*.midi"):
        midi = converter.parse(file)
        parse = None
        notes = []
        duration = []

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

        data = {f"music_{music_num}" : {"notes": notes, "durations": duration}}
        with open("notes_durations.json", "+a") as json_file:
            json.dump(data, json_file, indent=1)
        music_num += 1


generate_notesArray()