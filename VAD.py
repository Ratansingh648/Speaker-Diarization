import librosa
import numpy as np
from scipy.ndimage.morphology import binary_dilation
import struct
import webrtcvad

INT16_MAX = (2**15)-1


class VAD(object):

    def __init__(self, window_length=30, moving_average_width=8, max_silence_length=6, base_sampling_frequency=16000):
        self.vad_window_length = window_length
        self.vad_moving_average_width = moving_average_width
        self.vad_max_silence_length = max_silence_length
        self.base_sampling_frequency = base_sampling_frequency

    def _moving_average(self, array, width):
        """Smoothing Voice Detectoion with moving average"""
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        cumulative_sum = np.cumsum(array_padded, dtype=float)
        cumulative_sum[width:] -= cumulative_sum[:-width]
        return cumulative_sum[width-1:]/width

    def _normalize_volume(self, wav, target_dbfs, increase_only=False, decrease_only=False):
        """Sets the volume of audio file to certain target DB Level"""
        if increase_only and decrease_only:
            raise ValueError("Both Increase only and Decrease Only cannot be set together")

        rms = np.sqrt(np.mean((wav*INT16_MAX)**2))
        wav_dbfs = 20*np.log10(rms/INT16_MAX)
        dbfs_change = target_dbfs - wav_dbfs

        if (dbfs_change < 0 and increase_only) or (dbfs_change > 0 and decrease_only):
            return wav
        return wav * (10 ** (dbfs_change/20))

    def detect_voice(self, wav, sampling_frequency, aggression_mode=3, trim=False):
        """Detects the segments of voice"""

        wav = self._normalize_volume(wav, -30, increase_only=True)
        wav = librosa.resample(wav, orig_sr=sampling_frequency, target_sr=self.base_sampling_frequency)
        sampling_frequency = self.base_sampling_frequency
        samples_per_window = (self.vad_window_length*sampling_frequency)//1000
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]
        pcm_wav = struct.pack("%dh" % len(wav), *(np.round(wav * INT16_MAX)).astype(np.int16))

        voice_flags = []
        vad = webrtcvad.Vad(mode=aggression_mode)
        # Creating voice windows
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wav[window_start*2: window_end*2], sample_rate=sampling_frequency))
        voice_flags = np.array(voice_flags)

        # Running Moving average to Smooth the region
        audio_masks = self._moving_average(voice_flags, self.vad_moving_average_width)
        audio_masks = np.round(audio_masks).astype(np.bool)

        # Dilating the mask
        audio_masks = binary_dilation(audio_masks, np.ones(self.vad_max_silence_length + 1))
        audio_masks = np.repeat(audio_masks, samples_per_window)

        if trim:
            return audio_masks, wav[audio_masks]
        return audio_masks, wav
