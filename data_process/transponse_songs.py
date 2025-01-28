import sys
import json
import music21 as m21

data = {}
nData = {}
with open(sys.argv[1],'r') as file:
    data = json.load(file)

for music_id, music_content in data.items():
    nData[music_id] = []

    notes = music_content['notes']
    key = m21.key.Key(music_content["key"]["tonic"],music_content["key"]["mode"])
    target_tonic = m21.pitch.Pitch("C") if key.mode == "major" else m21.pitch.Pitch("A")
    interval = m21.interval.Interval(key.tonic,target_tonic)

    transposed_notes = []
    for note_str,duration in notes:
        if note_str != "Rest" and note_str != "":
            current_note = m21.note.Note(note_str)
            transposed_note = current_note.transpose(interval)
            transposed_notes.append((transposed_note.pitch.midi,duration))
        elif note_str == "Rest":
            transposed_notes.append((note_str, duration))
    nData[music_id] = transposed_notes
   
with open(sys.argv[2],'w') as file:
    json.dump(nData,file,indent=1)
