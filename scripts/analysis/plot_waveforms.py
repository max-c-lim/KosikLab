import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure

UNIT_ID = 180

FOLDER_PATH = "data/2953_centered_waveforms"
WAVEFORMS_PATH = FOLDER_PATH + f"/waveforms_{UNIT_ID}.npy"
TEMPLATES_PATH = FOLDER_PATH + "/templates.npy"
CHANNEL_MAP_PATH = FOLDER_PATH + "/channel_map.npy"

USE_POS_PEAK = False
SAMPLING_FREQUENCY = 20000


def plot_waveform(subplot: axes.Axes, waveform: np.array, sampling_frequency=None, x_window=None,
                  **plot_kwargs):
    """
    Plots waveform on subplot

    Parameters
    ----------
    subplot: axes.Axes
        Subplot to plot waveform on
    waveform: np.array
        shape: (n_samples,)
        peak is at index n_samples//2
    sampling_frequency: None or float
        If not None, use to convert time (x axis label) from samples to ms
    x_window: None or float or int
        If not None, plot waveform within (0, x_window)
    plot_kwargs:
        kwargs for .plot
    """
    x_cords = np.arange(waveform.size)
    peak_ind = waveform.size//2
    x_label = "Time Relative to Trough"

    x_cords -= peak_ind
    if sampling_frequency is not None:
        x_cords = x_cords / sampling_frequency * 1000

        x_label += " (ms)"

    if x_window is not None:
        if sampling_frequency is not None:
            x_window = x_window / sampling_frequency * 1000
        subplot.set_xlim(-x_window//2, x_window//2)

    subplot.plot(x_cords, waveform, **plot_kwargs)
    subplot.set_xlabel(x_label)


def main():
    waveforms = np.load(WAVEFORMS_PATH)
    ks_templates_all = np.load(TEMPLATES_PATH)
    ks_template = ks_templates_all[UNIT_ID, :, :]
    channel_map = np.load(CHANNEL_MAP_PATH)

    n_waveforms, n_samples, n_channels = waveforms.shape

    if USE_POS_PEAK:
        get_extremum_value = np.max
        get_extremum_ind = np.argmax
    else:
        get_extremum_value = np.min
        get_extremum_ind = np.argmin

    mean_template = waveforms.mean(axis=0)
    mean_template_chan = get_extremum_ind(get_extremum_value(mean_template, axis=0))

    ks_template_chan = get_extremum_ind(get_extremum_value(ks_template, axis=0))
    ks_template_chan_converted = channel_map[ks_template_chan]

    fig, a0 = plt.subplots(1)  # type: figure.Figure, axes.Axes
    plot_waveform(a0, ks_template[:, ks_template_chan]*100, sampling_frequency=SAMPLING_FREQUENCY,
                  x_window=n_samples)
    a0.set_xlim(-2.5, 2.5)

    plt.title(f"Unit ID: {UNIT_ID}")
    plt.show()


if __name__ == "__main__":
    main()

print("34 is positive peak")