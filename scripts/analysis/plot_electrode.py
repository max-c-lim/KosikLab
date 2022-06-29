from extract_mat import MatExtractor
from collections import OrderedDict
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

    units = mat.units

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
    # plt.rcParams["font.family"] = "Times New Roman"
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 200
    mat = MatExtractor("data/2953_sorted.mat")

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

    a00.scatter(channel_locations[:, 0], channel_locations[:, 1], [channel_dot_size]*n_channels, c="#000000")
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
    a01.set_ylabel(a00.get_ylabel())
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


# def create_electrode_plot(rec_name, results_path, matlab_dict):
#     Code for creating electrode plot automatically in spikesort_matlab4.py
#     ######################################################
#     ############  ELECTRODE PLOT PARAMETES  ##############
#     ######################################################
#     # Unit (circle) areas scale linearly with num_spikes
#
#     # Size of unit = num_spikes * ELECTRODE_SPIKE_SIZE_SCALE
#     ELECTRODE_SPIKE_SIZE_SCALE = 1
#     # Alpha value of units (0 = fully transparent, 1 = fully opaque)
#     ELECTRODE_SPIKE_ALPHA = 0.7
#     # Color of units (in hexadecimal)
#     ELECTRODE_SPIKE_COLOR = "#000000"
#     # For each value (n) in ELECTRODE_SPIKE_LEGEND_SCALES, a circle will appear on a legend
#     # that indicates the size of the unit if the unit had n spikes
#     ELECTRODE_SPIKE_LEGEND_SCALES = [100, 500, 1000]
#     # Size of dot when plotting electrode locations
#     ELECTRODE_DOT_SIZE = 1
#     # Color of dot when plotting electrode locations
#     ELECTRODE_DOT_COLOR = "#000000"
#     # Buffer for setting limits of x-axs
#     # The axis will go from -ELECTRODE_X_MAX_BUFFER to maximum_x_value_of_data + ELECTRODE_X_MAX_BUFFER
#     ELECTRODE_X_MAX_BUFFER = 100
#     # Buffer for setting limits of y-axs
#     # The axis will go from -ELECTRODE_Y_MAX_BUFFER to maximum_y_value_of_data + ELECTRODE_Y_MAX_BUFFER
#     ELECTRODE_Y_MAX_BUFFER = 100
#     # Label for x-axis
#     ELECTRODE_X_LABEL = "x (μm)"
#     # Label for y-axis
#     ELECTRODE_Y_LABEL = "y (μm)"
#
#     stopwatch = Stopwatch("CREATING ELECTRODE PLOT")
#     # Get plotting data
#     loc_sizes = OrderedDict()
#     loc_x = []
#     loc_y = []
#     for unit_dict in matlab_dict["units"]:
#         loc = (unit_dict["x_max"], unit_dict["y_max"])
#         size_increase = ELECTRODE_SPIKE_SIZE_SCALE * len(unit_dict["spike_train"])
#         if loc in loc_sizes:
#             loc_sizes[loc] += size_increase
#         else:
#             loc_sizes[loc] = size_increase
#             loc_x.append(loc[0])
#             loc_y.append(loc[1])
#
#     # Create plot
#     fig, ((a00, a01), (a10, a11)) = plt.subplots(2, 2)  # type:figure.Figure, ((axes.Axes, axes.Axes), (axes.Axes, axes.Axes))
#     fig.delaxes(a10)
#     fig.suptitle(f"Electrode Plots of {rec_name}")
#
#     # Plot recording MEA
#     channel_locations = matlab_dict["locations"]
#     x_max = max(channel_locations[:, 0])
#     y_max = max(channel_locations[:, 1])
#     n_channels = channel_locations.shape[0]
#     a00.scatter(channel_locations[:, 0], channel_locations[:, 1], [ELECTRODE_DOT_SIZE]*n_channels,
#                 c=ELECTRODE_DOT_COLOR)
#
#     a00.set_title(f"MEA - {n_channels} channels")
#     a00.set_xlabel(ELECTRODE_X_LABEL)
#     a00.set_ylabel(ELECTRODE_Y_LABEL)
#     a00.set_aspect("equal")
#     a00.set_xlim(-ELECTRODE_X_MAX_BUFFER, x_max+ELECTRODE_X_MAX_BUFFER)
#     a00.set_ylim(-ELECTRODE_Y_MAX_BUFFER, y_max+ELECTRODE_Y_MAX_BUFFER)
#
#     # Plot units
#     a01.scatter(loc_x, loc_y, loc_sizes, c=ELECTRODE_SPIKE_COLOR, alpha=ELECTRODE_SPIKE_ALPHA)
#     a01.set_title(f"Spike Locations")
#     a01.set_xlabel(ELECTRODE_X_LABEL)
#     a01.set_ylabel(ELECTRODE_Y_LABEL)
#     a01.set_aspect("equal")
#     a01.set_xlim(*a00.get_xlim())
#     a01.set_ylim(*a00.get_ylim())
#
#     # Plot size legend
#     loc_off_screen = (-1, -1)
#     a11.set_xlim(0, 0.1)
#     a11.set_ylim(0, 0.1)
#     a11.axis("off")
#     # Area scales linearly
#     for size in ELECTRODE_SPIKE_LEGEND_SCALES:
#         a11.scatter(*loc_off_screen, size * ELECTRODE_SPIKE_SIZE_SCALE,
#                     color=ELECTRODE_SPIKE_COLOR, alpha=ELECTRODE_SPIKE_ALPHA)
#     a11.legend(loc="center", labelspacing=2, borderpad=1)
#     fig.savefig(results_path / (rec_name+"_electrode_plots.png"))
#     fig.clf()
#     stopwatch.log_time("Done.")


if __name__ == "__main__":
    main()
