import sys
import json
import constants

import numpy as np
import tensorflow as tf
import music21 as m21
from music21 import instrument, note, stream, chord,converter
from fractions import Fraction


def extract_first_notes(midi_file):

    music = {}
    with open(midi_file,'r') as f:
        music = json.load(f)
    notes_len = 0
    first_key = next(iter(music))
    music_data = music[first_key]
    encoded_song = []
    for i  in range(constants.SEQUENCE_LEN):
        encoded_song.append(music_data[i])
    return encoded_song

def generate_notes(notes):

    music_note = []

    model = tf.keras.models.load_model(sys.argv[1])
    notes_label = {}
    with open(constants.NOTES_LABEL,'r') as f:
        notes_label = json.load(f)

    sequence_note = []
    for i in range(constants.SEQUENCE_LEN):
        sequence_note.append(notes_label['original'][notes[i]])
    # noteIn = np.reshape(sequence_note,network_input)
    # noteIn = noteIn/float(len(notes_label['original']))

    for _ in range(constants.MUSIC_LEN):
        noteOut = model.predict(np.array(sequence_note).reshape(1,-1))
        noteIndex = np.argmax(noteOut)
        print(noteIndex)

        noteResult = notes_label['reverse'][str(noteIndex)]

        music_note.append(noteResult)
        sequence_note.append(noteIndex)
        sequence_note = sequence_note[1:]
    return music_note

def create_midi(music_note):    
    offset = 0
    output_notes = []
    last_note = note.Note("C4")
    duration = 0

    midi_stream = stream.Stream()
    piano = instrument.Piano()
    midi_stream.append(piano)

    for i in range(min(constants.MUSIC_LEN, len(music_note))):  # Garante que não excedemos o tamanho
        m_note = music_note[i]
        if m_note == "Rest":  # Detecção de pausa
            last_note.duration.quarterLength = duration
            midi_stream.append(last_note)

            new_rest = note.Rest()
            new_rest.offset = offset
            last_note = new_rest
            duration = 0 
        elif m_note == "_":
            try:
                duration += constants.MUSIC_TIME_STEP
            except Exception as e:
                print(f"Erro em uma repeticao de nota {last_note} -> {e}")
        else:  # Detecção de nota simples
            try:
                last_note.duration.quarterLength = duration + constants.MUSIC_TIME_STEP
                print(f"Nota : {last_note.nameWithOctave}, duration:{last_note.duration.quarterLength}")
                midi_stream.append(last_note)

                new_note = note.Note(m_note)
                new_note.offset = offset
                last_note = new_note
                duration = 0
            except Exception as e:
                print(f"Erro ao processar nota simples: {m_note} -> {e}")

        offset += constants.MUSIC_TIME_STEP


    midi_stream.write('midi', fp='test_output.midi')
    print("Arquivo MIDI simples criado com sucesso!")



notes = extract_first_notes(sys.argv[2])
music_note = generate_notes(notes)
create_midi(music_note)