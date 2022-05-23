# Imports
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict


class MatExtractor:
    def __init__(self, path, max_unit_id=None):
        """
        Represents contents of loaded .mat file

        Parameters
        ----------
        :param path:
            Path to .mat file
        :param max_unit_id:
            If not None, exclude units with an id greater than given
        """
        self.name = "_".join(path.split("/")[-2:])
        self.dict = loadmat(path)
        self.units = self.get_units(max_unit_id)

    def _get_units_raw(self):
        # Returns array of units in struct form (raw form from scipy)
        return self.dict["units"][0]

    def get_unit(self, unit_id):
        """
        :param unit_id:
            id of unit
        :return:
            unit of unit_id as Unit class
        """
        units = self.get_units()
        for unit in units:
            if unit.get_id() == unit_id:
                return unit

    def get_units(self, max_unit_id=None):
        """
        Get units
        :param max_unit_id:
            If not None, exclude units with an id greater than given
        :return: An array of units as Unit class
        """
        units_raw = self._get_units_raw()
        units = []
        for i in range(len(units_raw)):
            unit = Unit(units_raw, i)
            if max_unit_id is not None:
                if unit.get_id() > max_unit_id:
                    continue
            units.append(Unit(units_raw, i))

        return units

    def get_unit_ids(self):
        # Get ids of the units
        ids = []
        for unit in self.units:
            ids.append(unit.get_id())
        return ids

    def get_spike_trains(self, time_start=None, time_end=None):
        """
        Get spike trains of the mat file

        :param time_start:
            If not None, exclude spikes that come before
        :param time_end:
            If not None, exclude spikes that come after

        :return:
        A list of numpy arrays that contain spike trains
        """
        units = self.units

        spike_trains = []
        for unit in units:
            unit_spike_train = unit.get_spike_train(time_start=time_start, time_end=time_end)

            spike_trains.append(unit_spike_train)

        return spike_trains


class Unit:
    """
    Represents a unit in the .mat file
    """

    def __init__(self, units_raw, index):
        self.dict = units_raw[index][0][0]

    def get_spike_train(self, time_start=None, time_end=None):
        spike_train = self.dict["spike_train"][0]
        if time_start is not None:
            indices = np.nonzero(spike_train >= time_start)
            spike_train = spike_train[indices]
        if time_end is not None:
            indices = np.nonzero(spike_train <= time_end)
            spike_train = spike_train[indices]
        return spike_train

    def get_x_max(self):
        return self.dict["x_max"][0][0]

    def get_y_max(self):
        return self.dict["y_max"][0][0]

    def get_template(self):
        # Alessio's/TJ's: shape (num_channels, num_timepoints)
        # Mine/New SI API: shape (num_timepoints, num_channels)
        # In other words, one template for each channel

        template = self.dict["template"]
        if template.shape[1] < template.shape[0]:
            # In general, there should be more channels than time points
            # Transpose template to ensure (n_timepoints, n_channels)
            template = template.transpose((1, 0))
        return template

    def get_id(self):
        return self.dict["unit_id"][0][0]

    def get_electrode(self):
        # Get electrode id (channel id) that is closest to neuron (best waveform)
        return self.dict["electrode"][0][0]

    def get_template_max(self):
        # Of all the templates (one template for each channel), get the template with the largest value at any point
        # This will correspond to the cleanest template since it will be the template recorded by the closest electrode

        templates = self.get_template()
        if templates.shape[1] > templates.shape[0]:
            # In general, there should be more channels than time points
            # Transpose template to ensure (n_channels, n_timepoints)
            templates = templates.transpose((1, 0))
        max_channel = templates.max(axis=1).argmax()
        return templates[max_channel, :]

    def plot_templates(self, plot_all=True):
        """
        Plots templates
        """
        template = self.get_template()

        if plot_all:
            plt.plot(template)
            plt.title(f"Unit ID: {self.get_id()}")
            plt.show()
        else:
            for channel_ind in range(template.shape[1]):
                plt.plot(template[:, channel_ind])
                plt.title(f"Unit ID: {self.get_id()}  |  Channel: {channel_ind}")
                plt.show()


def find_similar_units(mat_extractor1, mat_extractor2):
    for unit1 in mat_extractor1.get_units():
        train1 = unit1.get_spike_train()
        for unit2 in mat_extractor2.get_units():
            train2 = unit2.get_spike_train()
            if len(train2) > len(train1):
                continue

            train_tj_truncated = train1[:len(train2)]
            if np.allclose(train_tj_truncated, train2, rtol=0.01):
                print(unit1.get_id())
                print(train1)
                print(unit2.get_id())
                print(train2)


def main():
    # Directory of .mat files
    root_path = "/Users/maxlim/KosikLab/data/maxone/"
    path_tj = root_path + "tj/"
    path_mx = root_path + "mx/"

    # .mat files
    rec_num = 2950
    path_mat_tj = path_tj + f"{rec_num}_sorted.mat"
    path_mat_mx = "/Users/maxlim/KosikLab/data/maxone/mx/0uM_sorted.mat"
    # path_mat_mx = path_mx + f"{rec_num}_sorted.mat"

    # MatExtractors
    mat_tj = MatExtractor(path_mat_tj)
    mat_mx = MatExtractor(path_mat_mx)

    test = mat_tj.get_units()[0]
    plt.plot(test.get_template())
    plt.show()


if __name__ == "__main__":
    main()
