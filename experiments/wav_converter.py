import sys
from midi2audio import FluidSynth

# Caminho para o arquivo MIDI e o SoundFont (.sf2)
midi_file = sys.argv[1]
soundfont_file = "soundfont.sf2"  # Baixe um SoundFont, como GeneralUser.sf2

# Instancie o sintetizador com o SoundFont
fs = FluidSynth(soundfont_file)

# Gere o arquivo WAV
wav_file = sys.argv[2]
fs.midi_to_audio(midi_file, wav_file)

print(f"Música convertida e salva em {wav_file}")
