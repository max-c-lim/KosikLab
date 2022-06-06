import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


unit_id = 99



# waveforms_raw = np.load("data/waveform_tests/waveforms_raw/waveforms_99.npy")
# waveforms_r aw_avg = np.mean(waveforms_raw, axis=0)
# waveforms_raw_samples = np.load("data/waveform_tests/waveforms_raw/sampled_index_99.npy")
#
# waveforms_curated = np.load("data/waveform_tests/waveforms_curated/waveforms_99.npy")
# waveforms_curated_avg = np.mean(waveforms_curated, axis=0)
# waveforms_curated_samples = np.load("data/waveform_tests/waveforms_curated/sampled_index_99.npy")
#
# for channel_ind in range(waveforms_curated_avg.shape[1]):
#     fig, axs = plt.subplots(2, 1)
#     axs[0].plot(waveforms_raw_avg[:, channel_ind])
#     axs[1].plot(waveforms_curated_avg[:, channel_ind])
#     plt.show()


rez_mat = loadmat("data/2953_mine_on_alessio_dat_kilosort2/rez.mat")
rez = rez_mat["rez"]
wraw = rez["Wraw"][0, 0]
wraw = wraw.transpose((2, 1, 0))

n_units, n_timepoints, n_channels = wraw.shape
for unit in range(n_units):
    plt.plot(wraw[unit, :, :])
    plt.show()


# Curation
# My   AL
# 359  339  # Alessio's does not initially include units with 0 spikes
# 323  323  # Mine eliminates these 0-spike-units here (after firing rate curation)
# 251  250
# 244  220
