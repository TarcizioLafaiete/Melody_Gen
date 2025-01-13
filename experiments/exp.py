import json
import glob

from music21 import converter, instrument, note, chord

notes = []
duration = []
with open("notes_durations.json",'r') as file:
    settings = json.load(file)
    notes = settings['notes']
    duration = settings['durations']

    for a in notes:
        print(a)

    for b in duration:
        print(b)