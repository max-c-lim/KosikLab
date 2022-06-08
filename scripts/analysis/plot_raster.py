import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import numpy as np
from extract_mat import MatExtractor


def compare_raster_plots(recording, mat_tj, mat_mx, time_start=120000, time_end=245000):
    fig, axs = plt.subplots(2, 1, sharex="all", figsize=(10, 7))

    fig.suptitle(f"Raster Plots of {recording}")
    fig.supxlabel("Time")
    fig.supylabel("Units")
    fig.canvas.manager.set_window_title(f"{recording}_raster_plots")

    mat_tj.plot_raster(title=f"TJ's sorting. N Units: {len(mat_tj.get_units())}", time_start=time_start,
                       time_end=time_end, subplot=axs[0])
    mat_mx.plot_raster(title=f"Max's sorting. N Units: {len(mat_mx.get_units())}", time_start=time_start,
                       time_end=time_end, subplot=axs[1])
    plt.show()


def plot_raster_old(mat, subplot: axes.Axes, random_time=False, time_start=None, time_end=None, title=None):
    """
    Create a raster plot of the units from .mat file

    Parameters
    ----------
    mat: MatExtractor
        Contains data for raster plot
    subplot:
        Subplot to plot raster plot on
    random_time: bool
        Choose time_start and time_end randomly. Returns time_start and time_end
    time_start
        Starting time
    time_end
        Ending time
    title: string
    """

    if random_time:
        time_max = max([max(spike_train) for spike_train in mat.get_spike_trains()])
        window_size = 1e5
        time_start = np.random.randint(time_max+1)
        time_end = time_start + window_size

    spike_trains = mat.get_spike_trains(time_start=time_start, time_end=time_end)
    spike_trains = [train for train in spike_trains if len(train) > 0]  # Some units are not active in window
    colors1 = ['C{}'.format(i) for i in range(len(spike_trains))]
    subplot.eventplot(spike_trains, colors=colors1)
    subplot.set_title(title)
    subplot.set_ylim(0.5, len(spike_trains))
    subplot.set_xlabel(f"Time (samples, {mat.get_sampling_frequency()} Hz)")
    subplot.set_ylabel("Units")

    if random_time:
        return time_start, time_end


def plot_raster(mat, subplot: axes.Axes, random_time=False, time_start=None, time_end=None, use_ms=False, title=None):
    """
    Create a raster plot of the units from .mat file

    Parameters
    ----------
    mat: MatExtractor
        Contains data for raster plot
    subplot:
        Subplot to plot raster plot on
    random_time: bool
        Choose time_start and time_end randomly. Returns time_start and time_end
    time_start
        Starting time
    time_end
        Ending time
    use_ms: bool
        If True, for x-axis, use real time (ms) instead of samples
    title: string
        Title of plot
    """

    if random_time:
        time_max = max([max(spike_train) for spike_train in mat.get_spike_trains()])
        window_size = 1e5
        time_start = np.random.randint(time_max+1)
        time_end = time_start + window_size

    if use_ms:
        sampling_frequency = mat.get_sampling_frequency()
    else:
        sampling_frequency = None

    spike_trains = mat.get_spike_trains(time_start=time_start, time_end=time_end, sampling_frequency=sampling_frequency)
    spike_trains = [train for train in spike_trains if len(train) > 0]  # Some units are not active in window

    for i, st in enumerate(spike_trains):
        subplot.plot(st, np.ones_like(st)*i, marker='.', markersize=2,
                     ls='')
    subplot.set_title(title)
    subplot.set_ylim(0.5, len(spike_trains))

    if use_ms:
        xlabel = f"Time (ms)"
    else:
        xlabel = f"Time (samples, {mat.get_sampling_frequency()} Hz)"
    subplot.set_xlabel(xlabel)
    subplot.set_ylabel("Units")

    if random_time:
        return time_start, time_end


def main():
    mat = MatExtractor("data/maxone_2953_sorted.mat")

    fig, a0 = plt.subplots(1)
    time_start, time_end = plot_raster(mat, a0, random_time=True, use_ms=True)
    plt.show()


if __name__ == "__main__":
    main()
