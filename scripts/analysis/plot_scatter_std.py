######################################################
###################  DATA FOLDERS ####################
######################################################
# Folders containing waveforms as .npy files with name format: waveforms_{unit_id}.npy. Example: waveforms_22.npy
# waveforms .npy files must have shape (num_waveforms, num_samples, num_channels)
# Each folder should contain the waveforms of a single recording
waveforms_folders = [
    "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_2950/waveforms/waveforms_raw/waveforms",
    "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_2953/waveforms/waveforms_raw/waveforms",
    "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_2954/waveforms/waveforms_raw/waveforms",
    "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_2957/waveforms/waveforms_raw/waveforms",
    "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_5116/waveforms/waveforms_raw/waveforms",
    "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_5118/waveforms/waveforms_raw/waveforms",
]

# Name of each recording that corresponds to each folder in waveforms_folders
# If length of waveforms_folders is not equal to length of waveforms_folders_names,
# recordings will be given names L1, L2, L3...
waveforms_folders_names = [
    "2950",
    "2953",
    "2954",
    "2957",
    "5116",
    "5118",
]

# Path to save plotting data
saved_plotting_data_path = "/home/maxlim/SpikeSorting/data/test/220605/scatter_plot_data"


######################################################
##############  WAVEFORMS PARAMATERS  ################
######################################################
# If True, use the standard deviation at the peak.
# If False, use the standard deviation over a window around the peak
std_at_peak = True

# Time before/after peak to include in window for std calculation if std_at_peak=False
ms_before = 0.5
ms_after = 1.5

# Sampling frequency of waveforms (in Hz)
# If None, WAVEFORMS_MS_BEFORE and WAVEFORMS_MS_AFTER will be number of samples (rather than ms)
sampling_frequency = 20000.

# If True and template exists as .npy file, delete .npy file and recompute it
recompute_templates = False

######################################################
###############  PLOTTING PARAMATERS  ################
######################################################
# If True, save plotting data in saved_plotting_data_path
save_data = True

# If True, save plotting figure in saved_plotting_data_path
save_fig = True

# The possible color choices of the different recordings (waveforms from same recording have same color)
# If there are less recordings than colors, not all colors will be used
recording_colors = [
    "#f74343",  # red
    "#fccd56",  # yellow
    "#74fc56",  # green
    "#56fcf6",  # light blue
    "#1e1efa",  # dark blue
    "#fa1ed2",  # pink
]

# Number of units to plot for each recording (None-> all units are used)
max_units_per_recording = 50


# Threshold for curating units based on minimum number of spikes (smaller values are removed)
# Used to plot a vertical line (None-> no line is plotted)
min_spikes_threshold = 50

# Threshold for curating units based on max norm std (greater values are removed)
# Used to plot a horizontal line (None-> no line is plotted)
max_norm_std_threshold = 0.3


######################################################
######################  CODE  ########################
######################################################
# User parameters check
assert len(recording_colors) >= len(waveforms_folders), "'recording_colors' has a smaller length than 'waveforms_folders'"

# region Setup
def print_stage(text):
    text = str(text)

    num_chars = 60
    char = "="
    indent = int((num_chars - len(text)) / 2)

    print()
    print(num_chars * char)
    print(indent * " " + text)
    print(num_chars * char)


print_stage("IMPORTING LIBRARIES")
import time

_import_start = time.time()
# region Imports
import numpy as np
from os import remove as del_file
from os.path import exists as path_exists
import matplotlib.pyplot as plt
from pathlib import Path
import re  # RegEx
from tqdm import tqdm
# endregion
print(f"Done. Time: {time.time() - _import_start:.2f}s")


# endregion


class Stopwatch:
    def __init__(self):
        self._time_start = time.time()

    def log_time(self, text=None):
        if text is None:
            print(f"Time: {time.time() - self._time_start:.2f}s")
        else:
            print(f"{text} Time: {time.time() - self._time_start:.2f}s")


class WaveformFile:
    """
    Class for representing a waveform .npy file with contents of waveforms.
    Waveforms array must have shape (n_waveforms, n_samples, n_channels)
    """
    def __init__(self, file_path: Path, rec_name: str):
        file_path = file_path.absolute()
        self.file_path = str(file_path)

        self.unit_id = WaveformFile.get_unit_id(file_path.name)
        self.template_path = f"{saved_plotting_data_path}/{rec_name}/templates/template_{self.unit_id}"

        # Memory cache
        self._waveforms = None
        self._template = None

    def get_waveforms(self, cache=True):
        """
        Get waveforms from self.file_path or from memory

        Parameters
        ----------
        cache: bool
            If True, store waveforms in memory

        Returns
        -------
        np.array containing waveforms
        """
        if self._waveforms is not None:
            return self._waveforms

        # print("Loading waveforms from file")
        # stopwatch = Stopwatch()
        waveforms = np.load(str(self.file_path), mmap_mode="r")
        # stopwatch.log_time("Done loading waveforms.")
        if cache:
            self._waveforms = waveforms
        return waveforms

    def get_template(self, cache=True):
        """
        Get the template (mean waveform) with shape (n_samples, n_channels)

        Parameters
        ----------
        cache: bool
            If True, template is cached in memory
        Returns
        -------
        np.array
        """
        if recompute_templates:
            self._template = None
            if path_exists(self.template_path):
                del_file(self.template_path)

        if self._template is not None:
            return self._template

        if path_exists(self.template_path):
            template = np.load(self.template_path)
        else:
            template = np.mean(self.get_waveforms(), axis=1)

        if cache:
            self._template = template

        if save_data and not path_exists(self.template_path):
            create_folder_parent(self.template_path)
            np.save(self.template_path, template)

        return template

    def get_chan_max(self):
        """
        Get the max channel of the unit based on the template's max channel
        (The max channel for the template is treated as the max channel of each waveform)

        Returns
        -------
        The index of the max channel
        """
        template = self.get_template()
        return np.min(template, axis=0).argmin()

    def get_num_spikes_and_std_norm(self):
        """
        Gets the number of spikes

        Computes standard deviation of waveforms normalized (divided by) amplitude
        at peak or window (decided by user parameters)
            (Peak/amplitude is defined by negative peak)

        Returns
        -------
        Number of spikes
        Normalized standard deviation
        """

        chan_max = self.get_chan_max()
        waveforms_max = self.get_waveforms()[:, :, chan_max]

        waveforms_amps = np.abs(waveforms_max.min(axis=1))
        n_spikes = waveforms_amps.size
        amp_mean = np.mean(waveforms_amps)

        peak_ind = waveforms_max.argmin(axis=1)
        if std_at_peak:
            std = np.std(waveforms_max[np.arange(peak_ind.size), peak_ind])
        else:
            n_before = ms_to_samples(ms_before)
            n_after = ms_to_samples(ms_after)

            waveforms_windows = np.vstack([wf[ind - n_before:ind + n_after] for wf, ind in zip(waveforms_max, peak_ind)])
            std = np.std(waveforms_windows, axis=0).mean()

        return n_spikes, std / amp_mean

    @staticmethod
    def is_file(file_name):
        """
        Check whether a file based on its name contains waveforms for a unit
        Waveform files have name format of "waveforms_#.npy" where # is a positive integer
            Example: waveforms_32.npy

        Parameters
        ----------
        file_name: str
            Name of the file

        Returns
        -------
        bool
            True = file_name contains waveforms
            False = file_name does not contains waveforms
        """
        return re.match(r"^waveforms_\d+\.npy$", file_name)

    @staticmethod
    def get_unit_id(file_name):
        """
        Gets the unit id of the waveforms based on the file name containing the waveforms


        Parameters
        __________
        file_name: str
            The name of the waveforms .npy file (waveforms_{unit_id}.npy)

        Returns
        -------
        The unit id
        """

        start_idx = len("waveforms_")
        end = -len(".npy")
        return int(file_name[start_idx:end])


def ms_to_samples(ms):
    """
    Convert time in ms to number of samples using sampling_frequency

    Parameters
    ----------
    ms: float or int
        Time in ms to convert to number of samples

    Returns
    -------
    time in samples
    """
    return ms * sampling_frequency / 1000.


def create_folder(folder_path, parents=True, exist_ok=True):
    path = Path(folder_path)
    if not path.exists():
        path.mkdir(parents=parents, exist_ok=exist_ok)
        # print(f"Created folder: {folder_path.absolute()}")


def create_folder_parent(file_path):
    # Create the parent of file_path
    path = Path(file_path).parent
    path.mkdir(parents=True, exist_ok=True)


def get_plotting_data(waveforms_folders, waveforms_folders_names):
    """
    Get data for plotting the points

    Parameters
    ----------
    waveforms_folders: list
        List of folders containing waveforms
    waveforms_folders_names: list
        List of names for each folder in waveforms_folders

    Returns
    -------
    plotting_points: dict
        {recording_name, (nums_spikes, stds_norms)}
        len(nums_spikes) == len(stds_norms)
        nums_spikes = number of spikes for each unit
        stds_norms = normalized std for each unit
    max_num_spikes: int
        Maximum number of spikes of any unit (used for setting x-limit of plot)
    max_std_norm: float
        Maximum normalized std of any unit (used for setting y-limit of plot)
    """

    if len(waveforms_folders) != len(waveforms_folders_names):
        waveforms_folders_names = [f"L{i+1}" for i in range(len(waveforms_folders))]

    plotting_points = {}
    max_num_spikes = 0
    max_std_norm = 0
    print_stage("COMPUTING PLOTTING DATA FROM FOLDERS")
    stopwatch_total = Stopwatch()
    for folder, name in zip(waveforms_folders, waveforms_folders_names):
        folder = Path(folder)
        print(f"Folder: {folder.absolute()}")
        nums_spikes = []
        stds_norms = []

        waveforms_files_list = [f.name for f in folder.iterdir() if WaveformFile.is_file(f.name)]
        if max_units_per_recording is None or len(waveforms_files_list) <= max_units_per_recording:
            waveforms_files_list_rand = waveforms_files_list
        else:
            waveforms_files_list_rand = np.random.choice(waveforms_files_list,
                                                           size=max_units_per_recording,
                                                           replace=False)
        # Save which units were plotted
        # np.save(f"{saved_plotting_data_path}plotted_unit_ids")

        for file in tqdm(waveforms_files_list_rand, total=len(waveforms_files_list_rand)):
            if not WaveformFile.is_file(file):
                continue
            waveform = WaveformFile(folder / file, rec_name=name)

            # stopwatch_comp = Stopwatch()
            # print("Computing data")
            n_spikes, std_norm = waveform.get_num_spikes_and_std_norm()

            nums_spikes.append(n_spikes)
            max_num_spikes = max(max_num_spikes, n_spikes)

            stds_norms.append(std_norm)
            max_std_norm = max(max_std_norm, std_norm)
            # stopwatch_comp.log_time("Done computing data")

        plotting_points[name] = (nums_spikes, stds_norms)
    stopwatch_total.log_time("Done getting waveforms.")
    return plotting_points, max_num_spikes, max_std_norm


def plot_scatter_std(waveforms_folders, waveforms_folders_names):
    """
    Plots a scatter plot of the normalized standard deviations of units

    Parameters
    ----------
    waveforms_folders: list
        List of folders containing waveforms
    waveforms_folders_names: list
        List of names for each folder in waveforms_folders
    """
    plotting_points, max_num_spikes, max_std_norm = get_plotting_data(waveforms_folders, waveforms_folders_names)
    x_max = max_num_spikes + 300
    y_max = max_std_norm + 0.2

    print_stage("PLOTTING DATA")
    stopwatch = Stopwatch()
    for color, (rec_name, (nums_spikes, stds_norms)) in zip(recording_colors, plotting_points.items()):
        plt.scatter(nums_spikes, stds_norms, c=len(nums_spikes)*[color], label=rec_name)

    threshold_lines_kwargs = {"linestyle": "dotted", "linewidth": 1, "c": "#000000"}
    if min_spikes_threshold is not None:
        plt.axvline(min_spikes_threshold, **threshold_lines_kwargs)
        plt.text(min_spikes_threshold, y_max, min_spikes_threshold, horizontalalignment="center")
    if max_norm_std_threshold is not None:
        plt.axhline(max_norm_std_threshold, **threshold_lines_kwargs)
        plt.text(x_max, max_norm_std_threshold, max_norm_std_threshold, verticalalignment="center")

    plt.xlim(0, x_max)
    plt.xlabel("Number of spikes")

    plt.ylim(0, y_max)
    plt.ylabel("av. STD / amplitude")
    plt.legend()

    if save_fig:
        create_folder(saved_plotting_data_path)
        fig_path = saved_plotting_data_path + "/max_norm_std_plot.png"
        plt.savefig(fig_path)
        print(f"Saved plot to {fig_path}")

    plt.show()
    stopwatch.log_time("Done plotting data.")


def main():
    plot_scatter_std(waveforms_folders, waveforms_folders_names)


if __name__ == "__main__":
    main()
