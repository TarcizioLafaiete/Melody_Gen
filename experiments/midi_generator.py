import sys
import json
import constants

import numpy as np
import tensorflow as tf
import music21 as m21
from music21 import instrument, note, stream, chord,converter
from fractions import Fraction

def extract_first_100_notes(midi_file):

    # Carrega o arquivo MIDI
    midi = converter.parse(midi_file)
    notes = []
    duration = []

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
            symb = '.'.join(str(n) for n in element.normalOrder)
        elif isinstance(element, note.Rest):
            symb = "Rest"
        notes.append(symb)
        duration.append(f"{element.quarterLength}")

        # Limita a 100 elementos
        if len(notes) >= constants.SEQUENCE_LEN:
            break

    return notes,duration

def generate_notes(network_input, notes,duration):

    music_note = []
    music_duration = []

    model = tf.keras.models.load_model(sys.argv[1])
    notes_label = {}
    duration_label = {}
    with open(constants.NOTES_LABEL,'r') as f:
        notes_label = json.load(f)
    with open(constants.OFFSET_LABEL,'r') as f:
        duration_label = json.load(f)

    sequence_note = []
    sequence_duration = []
    for i in range(constants.SEQUENCE_LEN):
        sequence_note.append(notes_label['original'][notes[i]])
        sequence_duration.append(duration_label['original'][duration[i]])
    

    noteIn = np.reshape(sequence_note,network_input)
    durationIn = np.reshape(sequence_duration,network_input)

    for _ in range(constants.MUSIC_LEN):
        noteOut,durationOut = model.predict([noteIn,durationIn])
        noteIndex = np.argmax(noteOut)
        durationIndex = np.argmax(durationOut)

        noteResult = notes_label['reverse'][str(noteIndex)]
        durationResult = duration_label['reverse'][str(durationIndex)]

        music_note.append(noteResult)
        music_duration.append(durationResult)
        sequence_note.append(noteIndex)
        sequence_duration.append(durationIndex)
        sequence_note = sequence_note[1:]
        sequence_duration = sequence_duration[1:]

    return music_note,music_duration

def create_midi(music_note, music_duration):
    """Convert the output from the prediction to notes and create a MIDI file."""
    assert len(music_note) == len(music_duration), "Notas e durações devem ter o mesmo tamanho!"
    
    offset = 0
    output_notes = []

    for i in range(min(constants.MUSIC_LEN, len(music_note))):  # Garante que não excedemos o tamanho
        m_note = music_note[i]
        try:
            m_duration = m21.duration.Duration(float(Fraction(music_duration[i])))
        except Exception as e:
            print(f"Erro ao processar duração: {music_duration[i]} -> {e}")
            continue

        if ('.' in m_note) or m_note.isdigit():  # Detecção de acorde
            notes_in_chord = m_note.split('.')
            notes = []
            for current_note in notes_in_chord:
                try:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                except Exception as e:
                    print(f"Erro ao processar nota em acorde: {current_note} -> {e}")
            new_chord = chord.Chord(notes)
            new_chord.duration = m_duration
            new_chord.offset = offset
            output_notes.append(new_chord)
        elif m_note == "Rest":  # Detecção de pausa
            new_rest = note.Rest()
            new_rest.duration = m_duration
            new_rest.offset = offset
            output_notes.append(new_rest)
        else:  # Detecção de nota simples
            try:
                new_note = note.Note(m_note)
                new_note.duration = m_duration
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            except Exception as e:
                print(f"Erro ao processar nota simples: {m_note} -> {e}")

        # Atualizar offset
        offset += float(Fraction(music_duration[i]))

    print(output_notes)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='test_output.mid')
    print("Arquivo MIDI simples criado com sucesso!")



notes,duration = extract_first_100_notes(sys.argv[2])
music_note,music_duration = generate_notes((1,constants.SEQUENCE_LEN,1),notes,duration)
create_midi(music_note,music_duration)