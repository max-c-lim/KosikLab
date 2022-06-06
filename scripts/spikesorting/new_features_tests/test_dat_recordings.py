# TESTING ON 2950 CONTROL RECORDING (0uM)
# Test how different parallel processing parameters affects .dat file after bandpass filter

import time


class Stopwatch:
    def __init__(self):
        self._time_start = time.time()

    def log_time(self, text=None):
        if text is None:
            print(f"Time: {time.time() - self._time_start:.2f}s")
        else:
            print(f"{text} Time: {time.time() - self._time_start:.2f}s")


def print_stage(text):
    text = str(text).upper()

    num_chars = 60
    char = "="
    indent = int((num_chars - len(text)) / 2)

    print()
    print(num_chars * char)
    print(indent * " " + text)
    print(num_chars * char)


def write_dat(path, total_memory):
    BinaryRecordingExtractor.write_recording(rec_filtered, file_paths=path,
                                             dtype='int16', total_memory=total_memory,
                                             n_jobs=64, verbose=True, progress_bar=True)


print_stage("importing packages")
stopwatch = Stopwatch()
from spikeinterface.extractors import MaxwellRecordingExtractor, BinaryRecordingExtractor
from spikeinterface.toolkit.preprocessing import bandpass_filter
import numpy as np
# from spikeinterface import widgets as sw
stopwatch.log_time("Done importing.")

print_stage("loading raw recording and filtering")
stopwatch = Stopwatch()
rec_raw = MaxwellRecordingExtractor("/home/maxlim/SpikeSorting/data/maxone/200123/2950/recordings/0uM.raw.h5")
rec_filtered = bandpass_filter(rec_raw, freq_min=300, freq_max=6000)
stopwatch.log_time("Done.")

print_stage("Loading .dat recording files")
stopwatch = Stopwatch()
rec_folder = "/home/maxlim/SpikeSorting/wetai-gym/maxwell/tetanus/sanity_check/2950_test_recordings/"
path_G4 = rec_folder + "4G_total_64_jobs.dat"
path_G6 = rec_folder + "6G_total_64_jobs.dat"
path_G10 = rec_folder + "10G_total_64_jobs.dat"

g4 = BinaryRecordingExtractor(path_G4, sampling_frequency=rec_filtered.get_sampling_frequency(),
                              num_chan=rec_filtered.get_num_channels(), dtype=rec_filtered.get_dtype())
g6 = BinaryRecordingExtractor(path_G6, sampling_frequency=rec_filtered.get_sampling_frequency(),
                              num_chan=rec_filtered.get_num_channels(), dtype=rec_filtered.get_dtype())
g10 = BinaryRecordingExtractor(path_G10, sampling_frequency=rec_filtered.get_sampling_frequency(),
                               num_chan=rec_filtered.get_num_channels(), dtype=rec_filtered.get_dtype())
stopwatch.log_time("Done loading.")

print_stage("Getting traces")
stopwatch = Stopwatch()
traces_g4 = g4.get_traces()
traces_g6 = g6.get_traces()
traces_g10 = g10.get_traces()
all_traces = {"4": traces_g4, "6": traces_g6, "10": traces_g10}
stopwatch.log_time("Done.")

print_stage("running tests")
stopwatch = Stopwatch()
size = traces_g6.size
chunk_size = 45955
print(np.all(np.isclose(traces_g4[:chunk_size], traces_g6[:chunk_size])))
# Todo: Look at traces near the end of the chunk. Traces seem to be identical in beginning but different at chunk border
"""
When using 64 jobs...
TOTAL_MEMORY = 4G: 30637 chunk_size, 118 chunks, 36s total, 3.22it/s
TOTAL_MEMORY = 6G: 45955 chunk_size, 79 chunks, 39s total, 2.01it/s
TOTAL_MEMORY = 10G: 76593 chunk_size, 47 chunks, 48s, 1.02s/it
"""

# plt.plot(traces_g4[:200, 4])
# plt.savefig(rec_folder + "g4_0to200.png")
# plt.cla()
#
# plt.plot(traces_g6[:200, 4])
# plt.savefig(rec_folder+'g6_0to200.png')

stopwatch.log_time("Done plotting.")
