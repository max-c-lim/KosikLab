import numpy as np
import matplotlib.pyplot as plt


def get_channels_max(waveforms):
    if len(waveforms.shape) == 3:  # (n_waveforms, n_samples, n_chans)
        chans_max = waveforms.min(axis=1).argmin(axis=1)
        if not np.all(chans_max == chans_max[0]):
            print("WARNING: Not all waveforms have same max channel.")
            print(chans_max)
        return chans_max
    else:  # (n_samples, n_channels)
        return waveforms.min(axis=0).argmin()


def get_template(waveforms):
    return waveforms.mean(axis=0)


def get_template_max(waveforms):
    template = get_template(waveforms)
    chan_max = get_channels_max(template)
    return template[:, chan_max]


def get_template_chan_max(waveforms):
    template = get_template(waveforms)
    return get_channels_max(template)


def main():
    waveforms = np.load("data/waveforms_31.npy")
    n_waveforms, n_samples, n_channels = waveforms.shape
    chan_max = get_template_chan_max(waveforms)
    waveforms_max = waveforms[:, :, chan_max]
    # plt.plot(waveforms_max[1, :])
    plt.plot(np.mean(waveforms_max, axis=0))
    plt.show()


if __name__ == "__main__":
    main()
