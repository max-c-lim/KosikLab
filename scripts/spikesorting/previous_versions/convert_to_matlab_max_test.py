import spikeinterface.full as si

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


# matplotlib notebook

########################################################################

def save_to_mat(sorting, recording, waveform_extractor, file_name, save_electrode=False):
    from scipy.io import savemat
    file_name.parent.mkdir(exist_ok=True, parents=True)
    mdict = {"units": [], "locations": recording.get_channel_locations(), "fs": recording.get_sampling_frequency()}

    # get mx channels
    max_channels = si.get_template_extremum_channel(waveform_extractor)

    # get channel locations
    locations = recording.get_channel_locations()

    # get electrodes
    electrodes = recording.get_property('electrode')

    for u in sorting.get_unit_ids():

        unit_dict = {}

        max_channel_idx = recording.id_to_index(max_channels[u])

        spike_train = sorting.get_unit_spike_train(u)

        max_location = locations[max_channel_idx]

        template = waveform_extractor.get_template(u)

        unit_dict["spike_train"] = spike_train
        unit_dict["x_max"] = max_location[0]
        unit_dict["y_max"] = max_location[1]
        unit_dict["template"] = template
        unit_dict["unit_id"] = u
        if save_electrode:
            electrode = electrodes[max_channel_idx]
            unit_dict["electrode"] = electrode

        mdict["units"].append(unit_dict)
        print(f"\n\n{u}\n------------------")
        print(unit_dict)
    savemat(file_name, mdict)
    print(f"save to {file_name}")


########################################################################

overwrite = False

########################################################################

mat_file = Path("/home/maxlim/SpikeSorting/data/DL/sorted/si_2953.mat")
we = si.WaveformExtractor.load_from_folder("/home/maxlim/SpikeSorting/data/DL/sorted/processed/si/maxone_2953/cache/waveforms_curated")

# get recording
rec = we.recording

# get sorted units
sort_curated = we.sorting


print(f"N units: {len(sort_curated.get_unit_ids())}")

print(mat_file)
if not mat_file.is_file() or overwrite:
    print(sort_curated)
    print(rec)
    save_to_mat(sort_curated, rec, we, mat_file)


