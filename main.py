import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import sounddevice as sd
from spectralcluster import SpectralClusterer, RefinementOptions

from VAD import VAD
from VoiceEncoder import VoiceEncoder
from plot_speakers import plot_speakers


def create_labelling(labels, wav_splits):
    times = [((s.start + s.stop) / 2) / 16000 for s in wav_splits]
    labelling = []
    start_time = 0
    threshold = 0.25

    for i, time in enumerate(times):
        # If Speaker changes, append the last speaker and reset start time
        if i > 0 and labels[i] != labels[i-1]:

            # skip the last speaker for really short speak time
            # if (time-start_time) >= threshold:
            temp = [str(labels[i-1]), start_time, time]
            labelling.append(tuple(temp))
            start_time = time

        # For Last Segment, append the last data.
        if i == len(times)-1:
            temp = [str(labels[i]), start_time, time]
            labelling.append(tuple(temp))

    return labelling


def create_label_array(audio_mask, labelling, sample_len):
    array_sample = np.array([-1]*sample_len)
    label_array = []
    for label, start, end in labelling:
        label_array.extend([int(label)]*int(sampling_frequency*(end-start)))
    label_array.extend([int(label)]*(sum(audio_mask)-len(label_array)))
    array_sample[audio_mask] = -1*(np.array(label_array)+1)
    array_sample = np.multiply(audio_mask, array_sample)
    return array_sample


if __name__ == "__main__":
    #filename = "audio_with_noise.wav"
    #filename = "X2zqiX6yL3I.wav"
    filename = "2_people.wav"

    wav, sampling_frequency = librosa.load(os.path.join("data", filename), sr=None)

    vad = VAD()
    audio_mask, wav = vad.detect_voice(wav, sampling_frequency)
    sample_len = len(wav)
    wav2 = wav
    wav = wav[audio_mask]

    encoder = VoiceEncoder("cpu")

    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16, min_coverage=0.75)
    # speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]

    refinement_options = RefinementOptions(
        gaussian_blur_sigma=1,
        p_percentile=0.90)

    clusterer = SpectralClusterer(
        min_clusters=2,
        max_clusters=100,
        refinement_options=refinement_options)

    joblib.dump(cont_embeds, "cont_embed_{}.pkl".format(filename.split(".")[0]))

    labels = clusterer.predict(cont_embeds)
    labelling = create_labelling(labels, wav_splits)
    print(labelling)
    labeled = create_label_array(audio_mask, labelling, sample_len)

    #plt.plot(np.array(range(len(audio_mask)))/16000, labeled)
    #plt.plot(np.array(range(len(audio_mask)))/16000, audio_mask, "r--")
    #plt.plot(np.array(range(len(audio_mask)))/16000, wav2, color="black")
    # plt.show()

    print(np.array(labels)-2)
    plot_speakers(wav, wav_splits, 16000, np.array(labels)-2)
