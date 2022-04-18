import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sounddevice as sd
import sys
from time import sleep, perf_counter as timer


_default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_speakers(wav, wav_splits, sampling_rate, prediction, x_crop=5, show_time=False):
    fig, ax = plt.subplots()
    lines = [ax.plot([], [])[0]]
    text = ax.text(0, 0, "", fontsize=10)

    def init():
        ax.set_ylim(0, 5)
        ax.set_ylabel("Speaker ID")
        if show_time:
            ax.set_xlabel("Time (seconds)")
        else:
            ax.set_xticks([])
        ax.set_title("Speaker Diarization")
        return lines + [text]

    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    rate = 1 / (times[1] - times[0])
    crop_range = int(np.round(x_crop * rate))
    ticks = np.arange(0, len(wav_splits), rate)
    ref_time = timer()

    def update(i):
        # Crop plot
        crop = (max(i - crop_range // 2, 0), i + crop_range // 2)
        ax.set_xlim(i - crop_range // 2, crop[1])
        if show_time:
            crop_ticks = ticks[(crop[0] <= ticks) * (ticks <= crop[1])]
            ax.set_xticks(crop_ticks)
            ax.set_xticklabels(np.round(crop_ticks / rate).astype(np.int))

        # Plot the prediction
        pred = prediction[i]
        if pred != 0:
            message = "Speaker {}".format(abs(pred))
            color = _default_colors[abs(pred)]
        else:
            print(pred)
            message = "Unknown/No speaker"
            color = "Black"
        text.set_text(message)
        text.set_c(color)
        text.set_position((i, 0.96))

        # Plot data
        for line in lines:
            line.set_data(range(crop[0], i + 1), abs(prediction[crop[0]:i + 1]))

        # Block to synchronize with the audio (interval is not reliable)
        current_time = timer() - ref_time
        if current_time < times[i]:
            sleep(times[i] - current_time)
        elif current_time - 0.2 > times[i]:
            print("Animation is delayed further than 200ms!", file=sys.stderr)
        return lines + [text]

    ani = FuncAnimation(fig, update, frames=len(wav_splits), init_func=init, blit=not show_time,
                        repeat=False, interval=1)
    sd.play(wav, sampling_rate, blocking=False)
    plt.show()
