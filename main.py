from fileinput import filename
import numpy as np
import os
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

from VAD import VAD
from VoiceEncoder import VoiceEncoder

if __name__ == "__main__":
    filename = "audio_with_noise.wav"
    #filename = "X2zqiX6yL3I.wav"
    sampling_frequency, wav = read(os.path.join("data", filename))
    
    vad = VAD()
    audio_mask, wav = vad.detect_voice(wav, sampling_frequency)
    """
    fig1 = plt.figure(1)
    plt.plot(wav)
    plt.show()

    fig2 = plt.figure(2)
    plt.plot(np.multiply(audio_mask, wav))  
    plt.show()
    """
    encoder = VoiceEncoder("cpu")

    segments = [[0, 5.5], [6.5, 12], [17, 25]]
    speaker_names = ["Kyle Gass", "Sean Evans", "Jack Black"]
    speaker_wavs = [wav[int(s[0] * sampling_frequency):int(s[1] * sampling_frequency)] for s in segments]

    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16, min_coverage=0.75)
    speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
    print(np.array(speaker_embeds).shape)
    print(len(audio_mask))
    print(np.array(cont_embeds).shape)
    print(np.array(wav_splits).shape)