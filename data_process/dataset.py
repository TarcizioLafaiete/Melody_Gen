import sys
import json
import glob

from music21 import converter, instrument, note, chord

def generate_notesArray():

    music_num = 0
    data = {}

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
            duration.append(round(float(time_diff),3))

            if isinstance(element,note.Note):
                symb = str(element.pitch)
            elif isinstance(element,chord.Chord):
                symb = '.'.join(str(n) for n in element.normalOrder)
            elif isinstance(element,note.Rest):
                symb = "Rest"
            notes.append(f"{symb};{str(element.quarterLength)}")

        data[f"music_{music_num}"] = {"notes": notes, "offset": duration}
        with open(sys.argv[1], "w+") as json_file:
            json_file.write(json.dumps(data) + "\n")

        music_num += 1
    


generate_notesArray()