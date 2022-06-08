"""
Test how fast getting traces from MaxwellRecordingExtractor is
HYPOTHESIS: IF computing bandpass_filter lazily causes slowing traces retrieval,
            THEN, getting traces from recording without bandpass_filter will be much faster
"""

from spikeinterface.extractors import MaxwellRecordingExtractor, BinaryRecordingExtractor
from spikeinterface.toolkit.preprocessing import bandpass_filter
import time
from shutil import rmtree
from os.path import exists as path_exists
import numpy as np


class Stopwatch:
    def __init__(self, start_msg=None, use_print_stage=False):
        self._time_start = time.time()

        if start_msg is not None:
            if use_print_stage:
                pass
            else:
                print(str(start_msg))

    def log_time(self, text=None):
        if text is None:
            print(f"Time: {time.time() - self._time_start:.2f}s")
        else:
            print(f"{text} Time: {time.time() - self._time_start:.2f}s")


REC_PATH = "/home/maxlim/SpikeSorting/data/DL/recordings/maxone_2953.raw.h5"
SAVE_DATA_PATH = "/home/maxlim/SpikeSorting/data/test/220606"

N_JOBS = 64
TOTAL_MEMORY = "6G"
FREQ_MIN = 300
FREQ_MAX = 6000

start_frame = 13213
n_samples = int(5e4)
end_frame = start_frame + n_samples
traces_slice = slice(300, 600)

maxwell = MaxwellRecordingExtractor(REC_PATH)
print(type(maxwell))

# stopwatch = Stopwatch("Getting raw traces")
# traces = maxwell.get_traces(start_frame=start_frame, end_frame=end_frame)
# stopwatch.log_time("Done.")
# del traces

maxwell_filtered = bandpass_filter(maxwell, freq_min=FREQ_MIN, freq_max=FREQ_MAX)
stopwatch = Stopwatch("\nGetting traces after lazy bandpass")
traces = np.min(maxwell_filtered.get_traces(start_frame=start_frame, end_frame=end_frame)[traces_slice])
stopwatch.log_time("Done.")
del traces

# stopwatch = Stopwatch("\nWriting binary recording using lazy filtered recording")
# lazy_binary_path = SAVE_DATA_PATH + "/maxwell_2953_lazy_filtered.dat"
# if path_exists(lazy_binary_path):
#     rmtree(lazy_binary_path)
# BinaryRecordingExtractor.write_recording(maxwell_filtered, file_paths=lazy_binary_path,
#                                          dtype='int16', progress_bar=True,
#                                          n_jobs=N_JOBS, total_memory=TOTAL_MEMORY)
# stopwatch.log_time("Done.")

maxwell_filtered_saved_path = SAVE_DATA_PATH + "/maxwell_2953_filtered_cache"
if path_exists(maxwell_filtered_saved_path):
    rmtree(maxwell_filtered_saved_path)
stopwatch = Stopwatch("\nSaving bandpass filtered recording")
maxwell_filtered_saved = maxwell_filtered.save(folder=maxwell_filtered_saved_path, progress_bar=True,
                                               n_jobs=N_JOBS, total_memory=TOTAL_MEMORY)
stopwatch.log_time("Done.")
print(type(maxwell_filtered_saved))
stopwatch = Stopwatch("\nGetting traces from saved filtered recording")
traces = np.min(maxwell_filtered_saved.get_traces(start_frame=start_frame, end_frame=end_frame)[traces_slice])
stopwatch.log_time("Done.")
del traces

# stopwatch = Stopwatch("\nWriting binary recording using saved filtered recording")
# binary_path = SAVE_DATA_PATH + "/maxwell_2953_filtered.dat"
# if path_exists(binary_path):
#     rmtree(binary_path)
# BinaryRecordingExtractor.write_recording(maxwell_filtered_saved, file_paths=binary_path,
#                                          dtype='int16', progress_bar=True,
#                                          n_jobs=N_JOBS, total_memory=TOTAL_MEMORY)
# stopwatch.log_time("Done.")
