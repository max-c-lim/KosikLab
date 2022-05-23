from extract_mat import MatExtractor
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes


def get_electrode_data(mat, size_scale=1):
    """
    Gets data needed to plot electrodes (locations and sizes)

    Parameters
    ----------
    mat: MatExtractor
        Object containing .mat data
    size_scale: Union(int, float)
        Multiplying factor to increase circle size

    Returns
    -------
    loc_x: list
        x-locations of electrodes
    loc_y: list
        y-locations of electrodes
    list(loc_sizes.values): list
        Sizes for each electrode circle
    *All three lists are the same length
    """

    units = mat.get_units()

    loc_sizes = OrderedDict()
    loc_x = []
    loc_y = []
    for unit in units:
        loc = unit.get_x_max(), unit.get_y_max()
        size_increase = size_scale * len(unit.get_spike_train())
        if loc in loc_sizes:
            loc_sizes[loc] += size_increase
        else:
            loc_sizes[loc] = size_increase
            loc_x.append(loc[0])
            loc_y.append(loc[1])

    return loc_x, loc_y, list(loc_sizes.values())


def main():
    # Directory of .mat files
    path_data = "/Users/maxlim/KosikLab/data/maxone/"

    path_tj = path_data + "tj/"
    path_mx = path_data + "mx/"

    # .mat files
    rec_num = 2950
    path_mat_tj = path_tj + f"{rec_num}_sorted.mat"
    path_mat_mx = path_mx + f"{rec_num}_sorted.mat"

    # MatExtractors
    mat_tj = MatExtractor(path_mat_tj)
    mat_mx = MatExtractor(path_mat_mx)

    # Plotting Electrodes
    size_scale = 1
    electrode_data_tj = get_electrode_data(mat_tj, size_scale=size_scale)
    electrode_data_mx = get_electrode_data(mat_mx, size_scale=size_scale)

    alpha = 0.7
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"Electrode Plots of {rec_num}")
    fig.supxlabel("x")
    fig.supylabel("y")
    fig.canvas.manager.set_window_title(f"{rec_num}_electrode_plots")

    locations_all = (*electrode_data_tj[0], *electrode_data_tj[1], *electrode_data_mx[0], *electrode_data_mx[1])
    lim_max = max(locations_all)
    for ax in axs:
        for a in ax:  # type:axes.Axes
            a.set_aspect("equal")
            a.set_xlim(0, lim_max)
            a.set_ylim(0, lim_max)

    scatter_mx = axs[0, 0].scatter(*electrode_data_mx, c="#0000FF", alpha=alpha, label="Max")
    scatter_tj = axs[0, 0].scatter(*electrode_data_tj, c="#FF0000", alpha=alpha, label="TJ")
    axs[0, 0].set_title("Comparison")
    legend = axs[0, 0].legend((scatter_tj, scatter_mx),
               ("TJ", "MX"),
               loc="lower right")
    for i in range(len(legend.legendHandles)):
        legend.legendHandles[i]._sizes = [100]

    scatter_tj = axs[1, 0].scatter(*electrode_data_tj, c="#FF0000", alpha=alpha, label="TJ")
    scatter_mx = axs[1, 0].scatter(*electrode_data_mx, c="#0000FF", alpha=alpha, label="Max")
    axs[1, 0].set_title("Comparison")
    legend = axs[1, 0].legend((scatter_tj, scatter_mx),
               ("TJ", "MX"),
               loc="lower right")
    for i in range(len(legend.legendHandles)):
        legend.legendHandles[i]._sizes = [100]

    axs[0, 1].scatter(*electrode_data_mx, c="#0000FF", alpha=alpha, label="Max")
    axs[0, 1].set_title("Max's")

    axs[1, 1].scatter(*electrode_data_tj, c="#FF0000", alpha=alpha, label="TJ")
    axs[1, 1].set_title("TJ's")

    spacing = 0.2
    plt.subplots_adjust(wspace=spacing, hspace=spacing)
    plt.show()


if __name__ == "__main__":
    main()
