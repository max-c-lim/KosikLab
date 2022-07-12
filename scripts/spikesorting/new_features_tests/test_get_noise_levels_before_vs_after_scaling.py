from spikeinterface.extractors import MaxwellRecordingExtractor
from spikeinterface.toolkit.preprocessing import bandpass_filter, ScaleRecording
from spikeinterface.toolkit import get_noise_levels

rec = MaxwellRecordingExtractor("/home/maxlim/SpikeSorting/data/DL/recordings/2953.raw.h5")
scaled = ScaleRecording(rec, gain=rec.get_channel_gains(), offset=rec.get_channel_offsets(), dtype="float32")
filtered = bandpass_filter(scaled, freq_min=300, freq_max=6000)

print(get_noise_levels(filtered, return_scaled=False))

unscaled = ScaleRecording(filtered, offset=-rec.get_channel_offsets())
unscaled = ScaleRecording(unscaled, gain=1/rec.get_channel_gains())

print("-"*50)
print(get_noise_levels(unscaled, return_scaled=True))
