# File paths of curation_history.json for each recording
curation_history_file_paths = [
    "data/curation_histories/curation_history_2953.json",
    "data/curation_histories/curation_history_2954.json",
    "data/curation_histories/curation_history_2957.json",
    "data/curation_histories/curation_history_5116.json",
    "data/curation_histories/curation_history_5118.json"
]

# Names of each recording that corresponds to curation_history.json in curation_history_file_paths
# If None, recordings will be given names L1, L2, L3...
recording_names = [
    "maxone_2953"
]

import matplotlib.pyplot as plt
import json
import pandas as pd


def curation_barplot():
    # Get recording names
    if len(curation_history_file_paths) != len(recording_names):
        recording_names_corrected = [f"L{i}" for i in range(1, len(curation_history_file_paths)+1)]
    else:
        recording_names_corrected = recording_names

    # Get data
    n_initials = []
    n_finals = []
    names = []
    for curation_history_file_path, recording_name in zip(curation_history_file_paths, recording_names_corrected):
        with open(curation_history_file_path, "r") as fp:
            curation_history = json.load(fp)
        unit_ids_initial = curation_history["initial"]
        last_curation = list(curation_history.keys())[-1]
        unit_ids_final = curation_history[last_curation]

        n_initials.append(len(unit_ids_initial))
        n_finals.append(len(unit_ids_final))
        names.append(recording_name)

    # Plot data
    df = pd.DataFrame({'Total Units': n_initials,
                       'Selected Units': n_finals}, index=names)
    df.plot.bar(rot=0)  # rot = rotation of bar labels. i.e. rot=0=labels have 0-degree rotation
    plt.xlabel("Recording")
    plt.ylabel("Number of Units")
    plt.show()


if __name__ == "__main__":
    curation_barplot()
