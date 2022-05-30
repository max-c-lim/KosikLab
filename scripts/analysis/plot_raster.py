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


def plot_raster(mat, subplot: axes.Axes, random_time=False, time_start=None, time_end=None, title=None):
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
    spike_trains = [train for train in spike_trains if len(train) > 0]
    colors1 = ['C{}'.format(i) for i in range(len(spike_trains))]
    subplot.eventplot(spike_trains, colors=colors1)
    subplot.set_title(title)
    subplot.set_ylim(0.5, len(spike_trains))
    subplot.set_xlabel("Time (samples)")
    subplot.set_ylabel("Units")

    if random_time:
        return time_start, time_end


def main():
    # Directory of .mat files
    root_path = "./data/maxone/"
    path_tj = root_path + "tj/"
    path_mx = root_path + "mx/"

    # .mat files
    rec_num = 2950
    path_mat_tj = path_tj + f"{rec_num}_sorted.mat"
    path_mat_mx = "./data/maxone/mx/0uM_sorted.mat"
    # path_mat_mx = path_mx + f"{rec_num}_sorted.mat"

    # MatExtractors

    mat_150 = MatExtractor("/Users/maxlim/KosikLab/scripts/analysis/data/maxone/mx/150_maxone_2953_sorted.mat")
    mat_300 = MatExtractor("/Users/maxlim/KosikLab/scripts/analysis/data/maxone/mx/300_maxone_2953_sorted.mat")

    fig, (a0, a1) = plt.subplots(2)
    time_start, time_end = plot_raster(mat_150, a0, random_time=True)
    plot_raster(mat_300, a1, time_start=time_start, time_end=time_end)
    plt.show()


if __name__ == "__main__":
    main()
