import numpy as np
import matplotlib.pyplot as plt

data = np.fromfile("data/2953_mine_on_alessio_dat_kilosort2/temp_wh.dat")
whitening_mat = np.fromfile("data/2953_mine_on_alessio_dat_kilosort2/whitening_mat_inv.npy")
spike_clusters = np.load("data/2953_mine_on_alessio_dat_kilosort2/spike_clusters.npy").flatten()
unit_ids = np.unique(spike_clusters)
unit_id = 0 # np.random.choice(unit_ids)
gain = 6.29425

amplitudes = np.load("data/2953_mine_on_alessio_dat_kilosort2/amplitudes.npy").flatten()
spike_amplitudes = amplitudes[np.flatnonzero(spike_clusters == unit_id)]

templates = np.load("data/2953_mine_on_alessio_dat_kilosort2/templates.npy")
channel_map = np.load("data/2953_mine_on_alessio_dat_kilosort2/channel_map.npy").flatten()

cluster_Amplitudes = np.genfromtxt("data/2953_mine_on_alessio_dat_kilosort2/cluster_Amplitude.tsv", delimiter='\t')[1:, :]
template = templates[unit_id, :, :] * np.mean(spike_amplitudes) * gain

from scipy.io import loadmat

rez_mat = loadmat(str("data/2953_mine_on_alessio_dat_kilosort2/rez.mat"))
rez = rez_mat["rez"]
wraw = rez["Wraw"][0, 0]
wraw = wraw.transpose((2, 1, 0))
template_wraw = wraw[unit_id, :, :] * gain

waveforms = np.load("data/2953_mine_on_alessio_dat_kilosort2/waveforms_0.npy")
waveforms = waveforms[:, :, channel_map]
template_waveform = np.mean(waveforms, axis=0)

chan_ind = np.argmin(np.min(template_wraw, axis=0))
fig, axs = plt.subplots(3, 1)
axs[0].plot(template_waveform[:, chan_ind])
axs[0].set_title(f"Waveform: {np.min(template_waveform[:, chan_ind])}")
axs[1].plot(template[:, chan_ind])
axs[1].set_title(f"Template: {np.min(template[:, chan_ind])}")
axs[2].plot(template_wraw[:, chan_ind])
axs[2].set_title(f"Template Wraw: {np.min(template_wraw[:, chan_ind])}")
plt.show()

exit()
for chan_ind in range(template_waveform.shape[1]):
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(template_waveform[:, chan_ind])
    axs[0].set_title(f"Waveform: {np.min(template_waveform[:, chan_ind])}")
    axs[1].plot(template[:, chan_ind])
    axs[1].set_title(f"Template: {np.min(template[:, chan_ind])}")
    axs[2].plot(template_wraw[:, chan_ind])
    axs[2].set_title(f"Template Wraw: {np.min(template_wraw[:, chan_ind])}")
    plt.show()

