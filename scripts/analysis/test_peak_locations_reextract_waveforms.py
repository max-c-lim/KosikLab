import numpy as np
import matplotlib.pyplot as plt
from extract_mat import MatExtractor
from spikeinterface.extractors import MaxwellRecordingExtractor
from spikeinterface.toolkit.preprocessing import bandpass_filter


def get_peak_index(waveform: np.array):
    # waveform is 1d and has shape (n_samples,)
    peak = np.min(waveform)
    peak_indices = np.flatnonzero(waveform == peak)
    return peak_indices[peak_indices.size//2]


def main():
    mat = MatExtractor("data/maxone_2953_sorted.mat")
    rec_raw = MaxwellRecordingExtractor("path")
    rec_filtered = bandpass_filter(rec_raw, freq_min=300, freq_max=6000)

    unit = mat.get_unit(22)
    chan_max = unit.get_chan_max()
    chan_id = rec_filtered.channel_ids[chan_max]
    traces = rec_filtered.get_traces(start_frame=0, end_frame=0, channel_ids=[chan_id], return_scaled=True).flatten()

    waveforms = []
    nbefore = nafter = 50
    for st in unit.get_spike_train():
        window = traces[st-nbefore:st+nafter]
        peak_index = get_peak_index(window)
        waveforms.append(traces[peak_index-nbefore:peak_index+nafter])
    waveforms = np.vstack(waveforms)
    np.save("data/test_waveforms_22.npy", waveforms)


if __name__ == "__main__":
    main()
