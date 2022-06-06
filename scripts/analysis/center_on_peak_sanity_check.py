# Imports
from scipy.io import loadmat
import numpy as np
from spikeinterface.extractors import MaxwellRecordingExtractor
from spikeinterface.toolkit.preprocessing import bandpass_filter
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure


# region extract_mat.py
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
# endregion


def cache_recording_filtered():
    recording = MaxwellRecordingExtractor(path_recording_raw)
    rec_filtered = bandpass_filter(recording, freq_min=300, freq_max=6000)
    print("Saving filtered recording")
    rec_filtered.save(folder=path_recording_cache, verbose=True)


def load_recording_filtered():
    return MaxwellRecordingExtractor.load_from_folder(path_recording_cache)


path_recording_raw = "/home/maxlim/SpikeSorting/data/DL/recordings/maxone_2953.raw.h5"
path_recording_cache = "data/maxone_2953_filtered"

recording = load_recording_filtered()  # type: MaxwellRecordingExtractor
mat = MatExtractor("data/maxone_2953_sorted.mat")

test = mat.get_unit(31)
spike_train = test.get_spike_train()
chan_max = test.get_chan_max()

buffer = 10

total = 0
for time in spike_train:
    start_frame = int(time - buffer)
    end_frame = int(time + buffer)

    window = recording.get_traces(start_frame=start_frame, end_frame=end_frame)[:, chan_max]
    correct_value = window.min()
    correct_indices = np.flatnonzero(window == correct_value)
    correct = correct_indices[correct_indices.size//2] + start_frame
    if correct != time:
        print(correct, time)
        fig, a0 = plt.subplots(1)  # type:figure.Figure, axes.Axes
        x_values = np.arange(start_frame, end_frame)
        a0.plot(x_values, window)
        a0.set_xticks(x_values)
        a0.axvline(time)
        plt.show()
    else:
        total += 1


print("-"*50)
print(total)

