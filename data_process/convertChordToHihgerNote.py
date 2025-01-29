import sys
import json

from music21 import chord, pitch

data = {}
nData = {}
with open(sys.argv[1],'r') as file:
    data = json.load(file)

for key in data:
    nData[key] = {'notes' : []}
    for notes in data[key]['notes']:
        if'.' in notes[0]:
            pitch_notes = [pitch.Pitch(n) for n in notes[0].split('.')]
            top_notes = sorted(pitch_notes,key=lambda p: p.midi,reverse=True)[:3]
            higher_notes = '.'.join(n.nameWithOctave for n in top_notes)
            nData[key]['notes'].append((higher_notes,notes[1]))
        else:
            nData[key]['notes'].append(notes)
    nData[key]['key'] = data[key]['key']


with open(sys.argv[2],'w') as file:
    json.dump(nData,file,indent=1)
