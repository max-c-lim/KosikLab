# Imports
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


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

    def get_sampling_frequency(self):
        return self.dict["fs"].squeeze()

    def get_channel_locations(self):
        # Each row is the (x, y) of a channel
        return self.dict["locations"]

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
        # Get the spike times of the unit

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

    def get_chan_max(self):
        """
        Get the channel with the largest amplitude
        (amplitude is based on negative peak)

        Returns
        -------
        Index of max channel
        """
        return self.get_template().min(axis=0).argmin()

    def get_template_max(self):
        # Of all the templates (one template for each channel),
        # get the template with the min value at any point (min value for amplitude defined as negative peak)
        # This will correspond to the cleanest template since it will be the template recorded by the closest electrode

        return self.get_template()[:, self.get_chan_max()]


def find_similar_units(mat_extractor1, mat_extractor2):
    """
    Finds which units in mat_extractor1 correspond to mat_extractor2

    Parameters
    ----------
    mat_extractor1: MatExtractor
    mat_extractor2: MatExtractor

    """
    similar_total = 0
    from tqdm import tqdm
    for unit1 in tqdm(mat_extractor1.get_units()):  # type: Unit
        st1 = unit1.get_spike_train()
        electrode1 = unit1.get_electrode()
        for unit2 in mat_extractor2.get_units():  # type: Unit
            st2 = unit2.get_spike_train()
            electrode2 = unit2.get_electrode()
            if electrode1 != electrode2:
                continue

            similar = True
            rtol = 0.00001
            if st2.size < st1.size:
                for time in st2:
                    if not np.any(np.isclose(time, st1, rtol=rtol)):
                        similar = False
                        break
            elif st1.size < st2.size:
                for time in st1:
                    if not np.any(np.isclose(time, st2, rtol=rtol)):
                        similar = False
                        break
            else:
                similar = np.all(np.isclose(st1, st2, rtol=rtol))

            if similar:
                similar_total += 1
                print(f"Mat 1: {unit1.get_id()} | Mat 2: {unit2.get_id()}")
                plt.plot(unit1.get_template_max())
                plt.plot(unit2.get_template_max())
                plt.show()
                break
    print(similar_total)


def main():
    mat = MatExtractor("data/maxone_2953_sorted.mat")
    print(mat.get_channel_locations().shape)

if __name__ == "__main__":
    main()
