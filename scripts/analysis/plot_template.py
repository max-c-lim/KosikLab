from extract_mat import MatExtractor, Unit, find_similar_units
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.axes._axes as axes
import matplotlib.figure as figure


def plot_template(unit: Unit, subplot: axes.Axes):
    """
    Plots waveforms

    Parameters
    ----------
    unit: Unit object containing template to plot
    subplot: subplot to plot on
    """
    templates = unit.get_template_max()
    subplot.plot(templates)


def main():
    for unit in MatExtractor("data/maxone_2953_sorted.mat").get_units():
        fig, (ax0) = plt.subplots(1, sharex="all")  # type: (figure.Figure, axes.Axes)
        plot_template(unit, ax0)
        plt.show()


if __name__ == "__main__":
    main()
