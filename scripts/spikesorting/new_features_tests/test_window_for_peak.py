"""
Test to see if plausible:
For each unit, extract an appropriate window around Kilosort2's templates' peaks
to use for centering waveforms on peak. The window from Kilosort2's templates will
be around the peak from the last point on the left that is 0 to the first point on
the right that is 0
"""

import numpy as np
import matplotlib.pyplot as plt


templates_path = "/Users/maxlim/KosikLab/scripts/analysis/data/float32/templates.npy"
templates_all = np.load(templates_path)
chans_max_all = templates_all.min(axis=1).argmin(axis=1)

for i, chan in enumerate(chans_max_all):
    template = templates_all[i, :, chan]

    template_mid = template.size//2
    left_cut_ind = np.flatnonzero(np.isclose(template[:template_mid], 0))[-1]
    right_cut_ind = template.size-1 - left_cut_ind

    plt.plot(template)
    plt.axvline(left_cut_ind, color="#000000", linestyle="dotted")
    plt.axvline(right_cut_ind, color="#000000", linestyle="dotted")
    plt.show()


