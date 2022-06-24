"""
This is an example script that uses SpikeInterface to load traces
from a maxwell h5 recording file into a numpy.array
"""

# Import SpikeInterface
from spikeinterface.extractors import MaxwellRecordingExtractor

# Load recording
maxwell_recording = MaxwellRecordingExtractor("/home/maxlim/SpikeSorting/data/DL/recordings/maxone_2953.raw.h5")  # example path: recording.raw.h5

# Get all traces
# The returned np.array will have shape (n_frames, n_channels)
# CAUTION: This requires a large amount of RAM since entire recording will be loaded
all_traces = maxwell_recording.get_traces(return_scaled=True)  # "return_scaled=True" converts the traces to uV scale

# Get traces from a specific start frame and end frame
# Start is inclusive and end is exclusive
# The returned np.array will have shape (end_frame-start_frame, n_channels)
start_frame = 100
end_frame = 120
window_traces = maxwell_recording.get_traces(return_scaled=True,
                                             start_frame=start_frame, end_frame=end_frame)

# Get traces from the nth channel to the mth channel
# n is inclusive and m is exclusive
n = 50
m = 100
channel_traces = maxwell_recording.get_traces(return_scaled=True,
                                              start_frame=start_frame,
                                              end_frame=end_frame,
                                              channel_ids=maxwell_recording.get_channel_ids()[n:m])
# Alternatively, you can get the traces from specific channels from the returned np.array
channel_traces = window_traces[:, n:m]

# Get channel locations
# The returned np.array will have shape (n_channels, 2)
# The 0th column contains the x-values of each channel
# The 1st column contains the y-values of each channel
channel_locations = maxwell_recording.get_channel_locations()
