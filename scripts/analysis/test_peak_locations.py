import numpy as np
import matplotlib.pyplot as plt

# Load waveforms
waveforms = np.load("data/waveforms_22.npy")
n_spikes, n_samples, n_chans = waveforms.shape

# Get max channel
template = np.mean(waveforms, axis=0)
max_chan = np.argmin(np.min(template, axis=0))

# Sort waveforms based on amplitude
indices = np.arange(n_spikes).tolist()  # type: list
amplitudes = np.min(waveforms, axis=(1, 2))
indices.sort(reverse=True, key=lambda index: amplitudes[index])


def get_peak_index(waveform: np.array):
    # waveform is 1d and has shape (n_samples,)
    peak = np.min(waveform)
    peak_indices = np.flatnonzero(waveform == peak)
    return peak_indices[peak_indices.size//2]


def plot_all():
    # Plot all waveforms at once
    waveforms_max = waveforms[indices, :, max_chan]

    fig, (a0, a1) = plt.subplots(2)

    # Plot waveforms centered using Kilosort2's spike times
    a0.set_title("Centered using Kilosort2's Spike Times")
    a0.plot(waveforms_max.transpose((1, 0)))

    # Plot waveforms centered on actual negative peak
    a1.set_title("Centered using Calculated Negative Peak")
    a1.set_xlim(0, waveforms_max.shape[1])
    nbefore = nafter = 15
    for wave in waveforms_max:
        peak_ind = get_peak_index(wave)
        wave_recentered = wave[peak_ind - nbefore:peak_ind + nafter]
        left_pad = (wave.size - wave_recentered.size)//2
        a1.plot(np.arange(left_pad, left_pad+wave_recentered.size), wave_recentered)

    plt.show()


def plot_iteratively():
    # Plot each waveform one by one
    for i in indices:
        wave = waveforms[i, :, max_chan]

        plt.title("Compare Different Centering Methods")
        plt.plot(np.arange(wave.size), wave, label="Kilosort2")

        peak_ind = get_peak_index(wave)
        offset = wave.size//2 - peak_ind
        plt.plot(np.arange(wave.size)+offset, wave, label="Peak")

        plt.legend(loc="lower right")
        plt.show()


plot_all()
