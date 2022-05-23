from extract_mat import MatExtractor
import numpy as np
import matplotlib.pyplot as plt


def curate_fr(units, fr_thresh, total_duration):
    units_removed_fr = []
    units_curated = []
    for unit in units:
        n_spikes = unit.get_spike_train().size
        firing_rate = n_spikes / total_duration
        if firing_rate > fr_thresh:
            units_curated.append(unit)
        else:
            units_removed_fr.append(unit)

    return units_curated, units_removed_fr


def curate_isi(units, isi_viol_thresh, total_duration, fs):
    isi_threshold_ms = 1.5
    min_isi_ms = 0
    isi_threshold_s = isi_threshold_ms / 1000
    min_isi_s = min_isi_ms / 1000
    isi_threshold_samples = int(isi_threshold_s * fs)

    units_curated = []
    units_removed_isi = []
    for unit in units:
        spike_train = unit.get_spike_train()
        num_spikes = len(spike_train)
        isis = np.diff(spike_train)
        violation_num = np.sum(isis < isi_threshold_samples)
        violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s)
        total_rate = num_spikes / total_duration
        violation_rate = violation_num / violation_time
        violation_rate_ratio = violation_rate / total_rate

        if violation_rate_ratio < isi_viol_thresh:
            units_curated.append(unit)
        else:
            units_removed_isi.append(unit)

    return units_curated, units_removed_isi


def curate_units(units_curated, fr_thresh, isi_viol_thresh, num_cat, fs):
    if num_cat == 5:
        total_duration = 900.02
    else:
        total_duration = 179.88

    print(f"N units before curation: {len(units_curated)}")

    units_curated, units_removed_fr = curate_fr(units_curated, fr_thresh, total_duration)
    print(f"N units after firing rate curation: {len(units_curated)}")

    units_curated, units_removed_isi = curate_isi(units_curated, isi_viol_thresh, total_duration, fs)
    print(f"N units after ISI curation: {len(units_curated)}")

    return units_curated, units_removed_fr, units_removed_isi


def test_curation(mat_extractor):
    units_curated, units_removed_fr, units_removed_isi = curate_units(mat_extractor.get_units(), fr_thresh=0.05,
                                                                      isi_viol_thresh=0.3, num_cat=5, fs=20000)
    units_curated_ids = {unit.get_id() for unit in units_curated}
    units_removed_fr_ids = {unit.get_id() for unit in units_removed_fr}

    time_duration = 3.5e5
    time_start = np.random.random(1)[0] * max([np.max(train) for train in mat_extractor.get_spike_trains()])
    time_end = time_start + time_duration
    color_curated = "#0000FF40"
    color_fr = "#FF0000"
    color_isi = "#fc7703"
    length_curated = 1
    length_removed = 2
    spike_trains = []
    colors = []
    linelengths = []

    for unit in mat_extractor.get_units():
        unit_id = unit.get_id()
        if unit_id in units_curated_ids:
            color = color_curated
            width = length_curated
        else:
            width = length_removed
            if unit_id in units_removed_fr_ids:
                color = color_fr
            else:
                color = color_isi
        spike_trains.append(unit.get_spike_train(time_start=time_start, time_end=time_end))
        colors.append(color)
        linelengths.append(width)
    plt.eventplot(spike_trains, colors=colors, linelengths=linelengths)
    plt.title(mat_extractor.name)
    plt.get_current_fig_manager().set_window_title(mat_extractor.name.replace(".", "_"))

    plt.xlabel("Time (samples)")
    plt.ylabel("Units")
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=color_curated, lw=2),
                    Line2D([0], [0], color=color_fr, lw=2),
                    Line2D([0], [0], color=color_isi, lw=2)]

    plt.legend(custom_lines, [f'Curated: {len(units_curated)}', f'FR Viol: {len(units_removed_fr)}',
                              f'ISI Viol: {len(units_removed_isi)}'])
    plt.show()


def main():
    mat_tj = MatExtractor("./data/maxone/tj/2953_sorted.mat")
    test_curation(mat_tj)


if __name__ == "__main__":
    main()
