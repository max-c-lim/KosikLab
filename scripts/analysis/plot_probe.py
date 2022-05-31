from spikeinterface.extractors import MaxwellRecordingExtractor
from probeinterface.plotting import plot_probe
import matplotlib.pyplot as plt

recording = MaxwellRecordingExtractor("data/maxone_2953.raw.h5")
probe = recording.get_probe()
plot_probe(probe)
plt.show()
