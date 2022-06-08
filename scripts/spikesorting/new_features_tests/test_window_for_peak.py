"""
Test to see if plausible:
For each unit, extract an appropriate window around Kilosort2's templates' peaks
to use for centering waveforms on peak. The window from Kilosort2's templates will
be around the peak from the last point on the left that is 0 to the first point on
the right that is 0
"""

import numpy as np
import matplotlib.pyplot as plt


TEMPLATES_PATH = r"E:\KosikLab\code\scripts\analysis\data\templates.npy"

# If the positive peak of the waveform or template is POS_PEAK_THRESH times as large as
# the absolute value of negative peak, the positive peak is used for centering the waveform, amplitude,
# and finding the max channel. Otherwise, the negative peak is used
POS_PEAK_THRESH = 2

templates_all = np.load(TEMPLATES_PATH)

chans_neg_peaks_values = np.min(templates_all, axis=1)
chans_neg_peaks_indices = chans_neg_peaks_values.argmin(axis=1)
chans_neg_peaks_values = np.min(chans_neg_peaks_values, axis=1)

chans_pos_peaks_values = np.max(templates_all, axis=1)
chans_pos_peaks_indices = chans_pos_peaks_values.argmax(axis=1)
chans_pos_peaks_values = np.max(chans_pos_peaks_values, axis=1)

use_pos_peak = chans_pos_peaks_values >= POS_PEAK_THRESH * np.abs(chans_neg_peaks_values)

for i in range(templates_all.shape[0]):
    if i < 29:
        pass

    if use_pos_peak[i]:
        chan = chans_pos_peaks_indices[i]
        get_extremum_value = np.max
    else:
        chan = chans_neg_peaks_indices[i]
        get_extremum_value = np.min

    template = templates_all[i, :, chan]

    template_mid = template.size//2
    left_cut_ind = np.flatnonzero(np.isclose(template[:template_mid], 0))[-1]

    left_cut_ind = (template_mid - left_cut_ind) + (template_mid-left_cut_ind)/0.75

    right_cut_ind = template.size-1 - left_cut_ind

    plt.plot(template)
    plt.axvline(left_cut_ind, color="#000000", linestyle="dashed", label="Window Bounds")
    plt.axvline(right_cut_ind, color="#000000", linestyle="dashed")

    half_window_size = (template_mid - np.flatnonzero(np.isclose(template[:template_mid], 0))[-1])
    window_left = max(template_mid - half_window_size, 0)
    window_right = min(template_mid + half_window_size, template.size)
    window = slice(window_left, window_right)

    peak_value = get_extremum_value(template[window])
    peak_indices = np.flatnonzero(template[window] == peak_value)
    ind = peak_indices[peak_indices.size//2] - (window_right-window_left+1)//2
    st = template_mid
    st_centered = st + ind


    plt.axvline(st, color="#FF0000", linestyle="dotted", label="Kilosort Spike Time")
    plt.axvline(st_centered, color="#0000FF", linestyle="dotted", label="Centered Spike Time")

    plt.title(f"Index: {i}")
    plt.legend()
    plt.show()


