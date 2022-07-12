"""
To ensure data is consistent throughout the script,
the entire bandpass-filtered recording is saved
and reloaded, causing all subsequent get_traces() calls
for extracting waveforms are the same
"""

from spikeinterface.extractors import MaxwellRecordingExtractor
from spikeinterface.toolkit.preprocessing import bandpass_filter
import matplotlib.pyplot as plt
import numpy as np

start_frame = 1545
end_frame = start_frame + 100

raw = MaxwellRecordingExtractor("/home/maxlim/SpikeSorting/data/DL/recordings/2953.raw.h5")
raw = bandpass_filter(raw, freq_min=300, freq_max=6000, dtype="float32")
raw_traces = raw.get_traces(start_frame=start_frame, end_frame=end_frame, return_scaled=True)

cached = MaxwellRecordingExtractor.load_from_folder("/home/maxlim/SpikeSorting/data/dated/220704/2953/recording/2953_filtered")
cached_traces = cached.get_traces(start_frame=start_frame, end_frame=end_frame, return_scaled=False)

raw_chan = np.argmin(raw_traces[50, :])
cached_chan = np.argmin(cached_traces[50, :])

print(raw_traces)
print(raw_chan)
print("-"*50)
print(cached_traces)
print(cached_chan)

plt.plot(raw_traces[:, raw_chan], label="raw")
plt.plot(cached_traces[:, cached_chan], label="cached")
plt.legend()
plt.savefig("/home/maxlim/sz_scripts/testing.png")

