from extract_mat import MatExtractor
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure


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
    plt.rcParams["font.family"] = "Times New Roman"

    # Directory of .mat files
    data_path = "/Users/maxlim/KosikLab/data/maxone/"
    # MatExtractors
    rec_num = 2953
    # mat_tj = MatExtractor(data_path + f"tj/{rec_num}_sorted.mat")
    # mat_mx = MatExtractor(data_path + f"mx/{rec_num}_sorted.mat")
    mat_tj = MatExtractor("/Users/maxlim/KosikLab/scripts/good_experiment1_baseline_sorted.mat")
    mat_mx = MatExtractor("/Users/maxlim/KosikLab/scripts/experiment1_baseline_sorted.mat")

    # Plotting Electrodes
    size_scale = 1
    electrode_data_tj = get_electrode_data(mat_tj, size_scale=size_scale)
    electrode_data_mx = get_electrode_data(mat_mx, size_scale=size_scale)

    alpha = 0.7
    fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(2, 2)  # type:figure.Figure, ((axes.Axes, axes.Axes), (axes.Axes, axes.Axes))
    fig.suptitle(f"Electrode Plots of {rec_num}")
    fig.supxlabel("x location (μm)")
    fig.supylabel("y location (μm)")
    fig.canvas.manager.set_window_title(f"{rec_num}_electrode_plots")

    locations_all = (*electrode_data_tj[0], *electrode_data_tj[1], *electrode_data_mx[0], *electrode_data_mx[1])
    lim_max = max(locations_all)
    for ax in ((ax00, ax01), (ax10, ax11)):
        for a in ax:
            a.set_aspect("equal")
            a.set_xlim(0, lim_max)
            a.set_ylim(0, lim_max)

    ax00.set_title("Comparison")
    scatter_mx = ax00.scatter(*electrode_data_mx, c="#0000FF", alpha=alpha, label="Max")
    scatter_tj = ax00.scatter(*electrode_data_tj, c="#FF0000", alpha=alpha, label="TJ")
    legend_color = ax00.legend((scatter_tj, scatter_mx), ("TJ", "MX"), loc="upper left")
    for i in range(len(legend_color.legendHandles)):
        legend_color.legendHandles[i]._sizes = [100]

    loc_off_screen = (-1000, -1000)
    # size100 = ax00.scatter(*loc_off_screen, size_scale * 100, c="#000000")
    size1000 = ax00.scatter(*loc_off_screen, size_scale * 1000, c="#000000")
    # size2000 = ax00.scatter(*loc_off_screen, size_scale * 2000, c="#000000")
    legend_sizes = ax00.legend((size1000,),
                               ("1000 units",),
                               loc="upper right")

    ax00.add_artist(legend_color)
    ax00.add_artist(legend_sizes)
    scatter_tj = ax10.scatter(*electrode_data_tj, c="#FF0000", alpha=alpha, label="TJ")
    scatter_mx = ax10.scatter(*electrode_data_mx, c="#0000FF", alpha=alpha, label="Max")
    ax10.set_title("Comparison")
    legend = ax10.legend((scatter_tj, scatter_mx),
               ("TJ", "MX"),
               loc="upper left")
    for i in range(len(legend.legendHandles)):
        legend.legendHandles[i]._sizes = [100]

    ax01.scatter(*electrode_data_mx, c="#0000FF", alpha=alpha, label="Max")
    ax01.set_title("Max's")

    ax11.scatter(*electrode_data_tj, c="#FF0000", alpha=alpha, label="TJ")
    ax11.set_title("TJ's")

    spacing = 0.2
    plt.subplots_adjust(wspace=spacing, hspace=spacing)
    plt.show()


if __name__ == "__main__":
    main()
