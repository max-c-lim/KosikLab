import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure

UNIT_ID = 126

FOLDER_PATH = "data/waveforms/2953"
WAVEFORMS_PATH = FOLDER_PATH + f"/waveforms_{UNIT_ID}.npy"
TEMPLATES_PATH = FOLDER_PATH + "/templates.npy"
CHANNEL_MAP_PATH = FOLDER_PATH + "/channel_map.npy"

USE_POS_PEAK = False
SAMPLING_FREQUENCY = 20000


def plot_waveform(subplot: axes.Axes, waveform: np.array, sampling_frequency=None, x_window=None, y_label="Voltage (μV)", back_to_int=False,
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
        If not None, use to convert time (x-axis label) from samples to ms
    x_window: None or float or int
        If not None, plot waveform within (0, x_window)
    y_label: None or str
        If None, no label is used
    back_to_int: bool
        If True, convert waveform back to old way of handling recording (no scaling and int conversion)
    plot_kwargs:
        kwargs for .plot
    """
    x_cords = np.arange(waveform.size)
    peak_ind = waveform.size//2
    x_label = "Time Relative to Peak"

    x_cords -= peak_ind
    if sampling_frequency is not None:
        x_cords = x_cords / sampling_frequency * 1000

        x_label += " (ms)"
    else:
        x_label += " (samples)"

    if x_window is not None:
        if sampling_frequency is not None:
            x_window = x_window / sampling_frequency * 1000
        subplot.set_xlim(-x_window//2, x_window//2)
    if back_to_int:
        subplot.plot(x_cords, (waveform/6.295).astype(int)*6.295, **plot_kwargs)
    else:
        subplot.plot(x_cords, waveform, **plot_kwargs)
    subplot.set_xlabel(x_label)

    if y_label is not None:
        subplot.set_ylabel("Voltage (μV)")


def main():
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["font.size"] = 14

    unit_ids = (25, 30, 10, 93)

    fig, axs = plt.subplots(len(unit_ids), 3, figsize=(14, 8))  # type: figure.Figure, axes.Axes
    for si, UNIT_ID in enumerate(unit_ids):
        if UNIT_ID in {25, 30}:
            rec = "2954"
        else:
            rec = "2953"

        FOLDER_PATH = "data/waveforms/" + rec
        TEMPLATES_PATH = FOLDER_PATH + "/templates.npy"
        CHANNEL_MAP_PATH = FOLDER_PATH + "/channel_map.npy"

        a0 = axs[si, 0]
        a1 = axs[si, 1]
        a2 = axs[si, 2]
        waveforms = np.load(FOLDER_PATH + f"/waveforms_{UNIT_ID}.npy")
        ks_templates_all = np.load(TEMPLATES_PATH)
        ks_template = ks_templates_all[UNIT_ID, :, :]
        channel_map = np.load(CHANNEL_MAP_PATH).flatten()

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

        templates_std = np.load(f"{FOLDER_PATH}/templates_std.npy")
        templates_average = np.load(f"{FOLDER_PATH}/templates_average.npy")

        # Show STD Norm
        for i in range(n_waveforms):
            plot_waveform(a0, waveforms[i, :, mean_template_chan], sampling_frequency=SAMPLING_FREQUENCY,
                          x_window=n_samples, c="#bbbbbb", alpha=0.1, y_label="Voltage (μV)")
        plot_waveform(a0, mean_template[:, mean_template_chan], sampling_frequency=SAMPLING_FREQUENCY,
                      x_window=n_samples, c="black", linewidth=1.75)

        plot_waveform(a1, templates_std[UNIT_ID, :, mean_template_chan], sampling_frequency=SAMPLING_FREQUENCY,
                      x_window=n_samples, y_label=None, c="blue")
        # a1.set_ylabel("STD")

        plot_waveform(a2, templates_std[UNIT_ID, :, mean_template_chan]/np.abs(mean_template[100, mean_template_chan]), sampling_frequency=SAMPLING_FREQUENCY,
                      x_window=n_samples, y_label=None, c="#ff0000")
        # a2.set_ylabel("STD Scaled by Amplitude")
        a2.hlines(0.3, xmin=-2.5, xmax=2.5, colors="#fe7e10")
        fig.tight_layout(w_pad=3)

        subplots = (a0, a1, a2)
        if si < 2:
            a2.set_yticks([0.1, 0.2, 0.3])
        else:
            a2.set_yticks([0.1, 0.3, 0.5])
        for s in subplots:
            s.set_xlim(-2.5, 2.5)
            s.set_xticks([-2.5, 0, 2.5])

        # Compare integer and float waveforms
        # for i in range(n_waveforms):
        #     # if i not in {55, 57}:
        #     #     continue
        #
        #     fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 6))  # type: figure.Figure, axes.Axes
        #     plot_waveform(a1, waveforms[i, :, mean_template_chan], sampling_frequency=SAMPLING_FREQUENCY,
        #                   x_window=n_samples, c="blue")
        #     plot_waveform(a0, waveforms[i, :, mean_template_chan], sampling_frequency=SAMPLING_FREQUENCY,
        #                   x_window=n_samples, c="#8b0717", back_to_int=True)
        #     a0.set_xlim(-2.5, 2.5)
        #     a0.set_xticks([-2.5, 0, 2.5])
        #     a1.set_xticks([-2.5, 0, 2.5])
        #     a1.set_xlim(-2.5, 2.5)
        #     a0.set_title("Integer Data")
        #     a1.set_title("Float Data")
        #
        #     plt.show()

        # Compare centered to not centered
        # for i in range(n_waveforms):
        #     if i not in np.flatnonzero(np.load(f"data/waveforms/2953/{UNIT_ID}_offsets.npy")):
        #         continue
        #
        #     if i != 86:
        #         continue
        #
        #     fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 6))  # type: figure.Figure, axes.Axes
        #     centered = waveforms[i, :, mean_template_chan]
        #     not_centered = waveforms[i, :, mean_template_chan][1:]
        #     plot_waveform(a1, centered, sampling_frequency=SAMPLING_FREQUENCY,
        #                   x_window=n_samples-2, c="blue")
        #     plot_waveform(a0, not_centered, sampling_frequency=SAMPLING_FREQUENCY,
        #                   x_window=n_samples-2, c="#8b0717")
        #     a0.set_xlim(-2.5, 2.5)
        #     a0.set_xticks([-2.5, 0, 2.5])
        #     a1.set_xticks([-2.5, 0, 2.5])
        #     a1.set_xlim(-2.5, 2.5)
        #
        #     a0.set_title(not_centered[not_centered.size//2])
        #     y_min = min(not_centered)-3
        #     y_max = max(not_centered)+3
        #     a0.vlines(0, y_min, y_max, colors="black", linewidth=2, linestyles="dotted", label="Spike Location")
        #     a0.scatter(0, not_centered[not_centered.size//2], color="black", s=130)
        #     a0.set_ylim(y_min, y_max)
        #     a1.set_title(centered[centered.size//2])
        #     fig.canvas.manager.set_window_title(f"{UNIT_ID}_{i}")
        #     a1.vlines(0, y_min, y_max, colors="black", linewidth=2, linestyles="dotted", label="Spike Location")
        #     a1.scatter(0, centered[centered.size//2], color="black", s=130)
        #     a1.set_ylim(y_min, y_max)
        #     a0.legend(loc="upper right")
        #     a1.legend(loc="upper right")
        #     plt.show()

    plt.show()



if __name__ == "__main__":
    main()
