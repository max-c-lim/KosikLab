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

    for _ in range(10):
        print("FLIPPING Y-VALUES")
    y_max = max(loc_y)
    loc_y = [y_max - y for y in loc_y]

    return loc_x, loc_y, list(loc_sizes.values())


def main():
    # plt.rcParams["font.family"] = "Times New Roman"
    mat = MatExtractor("data/maxone_2953_sorted.mat")

    # Plotting Electrodes
    size_scale = 1
    electrode_data = get_electrode_data(mat, size_scale=size_scale)

    fig, ((a00, a01), (a10, a11)) = plt.subplots(2, 2)  # type:figure.Figure, ((axes.Axes, axes.Axes), (axes.Axes, axes.Axes))
    fig.delaxes(a10)
    fig.suptitle(f"Electrode Plots")
    fig.canvas.manager.set_window_title(f"electrode_plots")

    # Plot MEA
    channel_locations = mat.get_channel_locations()
    n_channels = channel_locations.shape[0]
    channel_dot_size = 1

    for _ in range(10):
        print("FLIPPING Y-VALUES")
    y_flipped = max(channel_locations[:, 1])-channel_locations[:, 1]

    a00.scatter(channel_locations[:, 0], y_flipped, [channel_dot_size]*n_channels, c="#000000")
    a00.set_title(f"Recording MEA - {n_channels} channels")
    a00.set_xlabel("x (μm)")
    a00.set_ylabel("y (μm)")
    a00.set_aspect("equal")
    lim_buffer = 100
    a00.set_xlim(-lim_buffer, max(*electrode_data[0])+lim_buffer)
    a00.set_ylim(-lim_buffer, max(*electrode_data[1])+lim_buffer)

    # Plot Units' Locations
    spike_color = "#0000FF"
    spike_alpha = 0.7
    a01.scatter(*electrode_data, c=spike_color, alpha=spike_alpha)
    a01.set_title(f"Spike Locations")
    a01.set_xlabel(a00.get_xlabel())
    a01.set_ylabel(a00.get_xlabel())
    a01.set_aspect("equal")
    a01.set_xlim(*a00.get_xlim())
    a01.set_ylim(*a00.get_ylim())

    # Plot Size Legend
    loc_off_screen = (-1, -1)
    a11.set_xlim(0, 0.1)
    a11.set_ylim(0, 0.1)
    sizes_kwargs = {"color": spike_color, "alpha": spike_alpha}

    # Area scales linearly
    size100 = a11.scatter(*loc_off_screen, size_scale * 100, **sizes_kwargs)
    size500 = a11.scatter(*loc_off_screen, size_scale * 500, **sizes_kwargs)
    size1000 = a11.scatter(*loc_off_screen, size_scale * 1000, **sizes_kwargs)

    a11.legend((size100, size500, size1000), ("100 spikes", "500 spikes", "1000 spikes"),
              loc="center",
              labelspacing=2,
              borderpad=1)
    plt.show()


    # Comparison
    # alpha = 0.7
    # fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(2, 2)  # type:figure.Figure, ((axes.Axes, axes.Axes), (axes.Axes, axes.Axes))
    # fig.suptitle(f"Electrode Plots of {rec_num}")
    # fig.supxlabel("x location (μm)")
    # fig.supylabel("y location (μm)")
    # fig.canvas.manager.set_window_title(f"{rec_num}_electrode_plots")
    #
    # locations_all = (*electrode_data_tj[0], *electrode_data_tj[1], *electrode_data_mx[0], *electrode_data_mx[1])
    # lim_max = max(locations_all)
    # for ax in ((ax00, ax01), (ax10, ax11)):
    #     for a in ax:
    #         a.set_aspect("equal")
    #         a.set_xlim(0, lim_max)
    #         a.set_ylim(0, lim_max)
    #
    # ax00.set_title("Comparison")
    # scatter_mx = ax00.scatter(*electrode_data_mx, c="#0000FF", alpha=alpha, label="Max")
    # scatter_tj = ax00.scatter(*electrode_data_tj, c="#FF0000", alpha=alpha, label="TJ")
    # legend_color = ax00.legend((scatter_tj, scatter_mx), ("TJ", "MX"), loc="upper left")
    # for i in range(len(legend_color.legendHandles)):
    #     legend_color.legendHandles[i]._sizes = [100]
    #
    # loc_off_screen = (-1000, -1000)
    # # size100 = ax00.scatter(*loc_off_screen, size_scale * 100, c="#000000")
    # size1000 = ax00.scatter(*loc_off_screen, size_scale * 1000, c="#000000")
    # # size2000 = ax00.scatter(*loc_off_screen, size_scale * 2000, c="#000000")
    # legend_sizes = ax00.legend((size1000,),
    #                            ("1000 units",),
    #                            loc="upper right")
    #
    # ax00.add_artist(legend_color)
    # ax00.add_artist(legend_sizes)
    # scatter_tj = ax10.scatter(*electrode_data_tj, c="#FF0000", alpha=alpha, label="TJ")
    # scatter_mx = ax10.scatter(*electrode_data_mx, c="#0000FF", alpha=alpha, label="Max")
    # ax10.set_title("Comparison")
    # legend = ax10.legend((scatter_tj, scatter_mx),
    #            ("TJ", "MX"),
    #            loc="upper left")
    # for i in range(len(legend.legendHandles)):
    #     legend.legendHandles[i]._sizes = [100]
    #
    # ax01.scatter(*electrode_data_mx, c="#0000FF", alpha=alpha, label="Max")
    # ax01.set_title("Max's")
    #
    # ax11.scatter(*electrode_data_tj, c="#FF0000", alpha=alpha, label="TJ")
    # ax11.set_title("TJ's")
    #
    # spacing = 0.2
    # plt.subplots_adjust(wspace=spacing, hspace=spacing)
    # plt.show()


if __name__ == "__main__":
    main()
