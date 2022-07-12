"""
Template script to spike sort axon recordings using SpikeInterface>=0.90.

Author: Alessio Buccino
"""

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
from spikeinterface.exporters import export_to_phy

import h5py
import shutil
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
from pprint import pprint
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import axon_velocity as av
from scipy.io import savemat
import os
from tqdm import tqdm

mpl.use("agg")


### Helper functions ###
def extract_one_full_template(i, temp_dict, locations_all, full_templates, shared_channels, verbose):
    """
    Extracts full template from axon scan

    Parameters
    ----------
    i: int
        Template index
    temp_dict: dict
        Dictionary with list of 'templates' and 'locations' for each config
    locations_all: np.array
        Array with locations of all channels (num_all_locations, 2)
    full_templates: np.array
        Array with full templates (num_templates, num_all_locations, num_channels, num_sampleS)
    shared_channels: np.array
        Array with cumulative number of configs per channel
    verbose: bool
        If True, output is verbose
    """
    if verbose:
        print(f"Extracting template {i}")
    templates = temp_dict['templates']
    locations = temp_dict['locations']
    locs_tuple = [tuple(loc) for loc in locations_all]
    for (temp, locs) in zip(templates, locations):
        # find location indexes
        loc_idxs = np.array([list(locs_tuple).index(tuple(loc)) for loc in locs])
        full_templates[i, loc_idxs] += temp
        shared_channels[i, loc_idxs] += np.ones(len(loc_idxs), dtype='int')
    shared_channels[i][shared_channels[i] == 0] = 1
    full_templates[i] = full_templates[i] / shared_channels[i][:, np.newaxis]


def get_wells_and_rec_names(file):
    with h5py.File(file) as f:
        well_names = list(f["wells"])
        rec_names = list(f["recordings"])
    return well_names, rec_names


def sorting_folder_to_index(sort_folder):
    if sort_folder.name.startswith("sorting_curated"):
        prefix = len("sorting_curated_")
    else:
        prefix = len("sorting_")
    return int(sort_folder.name[prefix:])


def get_sorting_folders_from_cache(cache_folder):
    sorting_folders = [p for p in cache_folder.iterdir() if 'sorting_curated_' in p.name and p.is_dir()]
    if len(sorting_folders) == 0:
        # auto curate False
        sorting_folders = [p for p in cache_folder.iterdir() if 'sorting_' in p.name and p.is_dir()]
    return sorting_folders

############################################
################  INPUTS ###################
############################################

# list of axon scan folders you want to spike sort
axon_scan_folders = ['/home/maxlim/SpikeSorting/data/axon_debugging/test_data']

# list of intermediate folder where tmp and output files are saved
intermediate_folders = ['/home/maxlim/SpikeSorting/data/axon_debugging/220705/test_data_sorted']

# list of output folders where final .mat files are saved
mat_folders = ['/home/maxlim/SpikeSorting/data/axon_debugging/220705/test_data_sorted/mat']

assert len(axon_scan_folders) == len(intermediate_folders) == len(mat_folders), "'axon_scan_folders' and 'intermediate_folders' and 'mat_folders' " \
                                                                                "should have the same length"

# sorter name
sorter = 'kilosort2'
ss.Kilosort2Sorter.set_kilosort2_path("/home/maxlim/SpikeSorting/Kilosort2")
print(ss.Kilosort2Sorter.kilosort2_path)

# folder containing libcompression.so
os.environ['HDF5_PLUGIN_PATH'] = '/home/maxlim/SpikeSorting/extra_libraries'

# sorter params
sorter_params = {"n_jobs_bin": 1, "total_memory": "4G"}

############################################
################  PARAMS ###################
############################################

# If True, and spike sorting output is present, it's deleted and resorted
recompute_sorting = False
recompute_splitting = False
recompute_curation = False
recompute_full_template = False
recompute_axon_velocities = False
recompute_mat_files = False

# If True, filtered data and sorted outputs are saved in a format that it's easy to retrieve (.pkl)
dump_recording = True
dump_sorting = True

# If True, exports to Phy
export_raw_to_phy = False
export_curated_to_phy = False

# If True, unit templates are plotted for all units
plot_unit_templates = False
plot_unit_axons = True

### FILTER params ###
freq_min = 300
fre_max = 6000

### AUTOMATIC CURATION ###
# If True, output is automatically curated using quality metrics (QC)
auto_curate = True

# Thresholds for automatic curations (if any is None, that autocuration is skipped
# ISI-violation ratio (greater values are removed)
isi_viol_thresh = 0.5
# firing rate (smaller values are removed)
fr_thresh = 0.05
# signal-to-noise ratio (smaller values are removed)
snr_thresh = 5

### OTHER PROCESSING PARAMS ###
# number of jobs to use
n_jobs = 1
# total RAM to use
total_memory = "4G"
# Number of spikes per unit to compute templates (None-> all spikes are used)
max_spikes_per_unit = None
# Number of channels to compute center of mass
num_channels_for_com = 30

## AXON TRACKING PARAMS ##
params = av.get_default_graph_velocity_params()

# change params
params['detect_threshold'] = 0.01
params['kurt_threshold'] = 0.3
params['peak_std_threshold'] = 1
params['init_delay'] = 0.2
params['upsample'] = 2
params['min_path_length'] = 100
params['min_path_points'] = 5

for (axon_scan_folder, intermediate_folder, mat_folder) in zip(axon_scan_folders, intermediate_folders, mat_folders):
    axon_scan_folder = Path(axon_scan_folder)
    intermediate_folder = Path(intermediate_folder)
    t_start_all = time.time()
    output_folder = intermediate_folder / 'sorted' / sorter
    cache_folder = intermediate_folder / 'cache'
    cache_folder.mkdir(parents=True, exist_ok=True)
    figures_folder = intermediate_folder / 'figures'
    # figures_units = figures_folder / 'units'

    rec_paths = [p for p in axon_scan_folder.iterdir() if p.suffix == '.h5']

    if len(rec_paths) > 1:
        print("Reading Maxwell from old version files")
        # sort recordings
        rec_ids = [int(rec.name.split('.')[0]) for rec in rec_paths]
        rec_paths = np.array(rec_paths)[np.argsort(rec_ids)]

        ### Find common set of electrodes ###
        electrode_ids = []
        channel_ids = []
        recordings = []
        for i, rec_path in enumerate(rec_paths):
            # Load recording
            try:
                print(f"Loading Maxwell recording: {rec_path.name}")
                recording = se.MaxwellRecordingExtractor(rec_path)
                print(recording)
            except:
                raise Exception(f"Could not open the provided file: {rec_path} with the MaxwellRecordingExtractor")

            electrode_ids.append(recording.get_property("electrode"))
            channel_ids.append(recording.get_channel_ids())
            recordings.append(recording)
    else:
        print("Reading Maxwell from new version files")
        # sort recordings
        recording_file = rec_paths[0]
        well_names, rec_names = get_wells_and_rec_names(recording_file)

        ### Find common set of electrodes ###
        electrode_ids = []
        channel_ids = []
        recordings = []
        for i, rec_name in enumerate(rec_names):
            # Load recording
            try:
                print(f"Loading Maxwell recording: {rec_name}")
                recording = se.MaxwellRecordingExtractor(recording_file, rec_name=rec_name)
                print(recording)
            except:
                raise Exception(
                    f"Could not open the provided rec name: {rec_name} with the MaxwellRecordingExtractor")

            electrode_ids.append(recording.get_property("electrode"))
            channel_ids.append(recording.get_channel_ids())
            recordings.append(recording)

    for i in range(len(recordings)):
        rec = recordings[i]
        if rec.has_scaled_traces():
            gain = rec.get_channel_gains()
            offset = rec.get_channel_offsets()
        else:
            gain = 1
            offset = 0
        recordings[i] = st.scale(rec, gain=gain, offset=offset, dtype="float32")

    fs = recordings[0].get_sampling_frequency()
    # find common set as intersection
    common_electrodes = set(electrode_ids[0])
    for elec in electrode_ids:
        common_electrodes = common_electrodes.intersection(elec)
    common_electrodes = list(common_electrodes)

    print(f"Number of common electrodes: {len(common_electrodes)}")
    # find channels corresponding to common set
    common_channel_ids = []
    for (chans, elecs) in zip(channel_ids, electrode_ids):
        elec_idxs = [list(elecs).index(el) for el in common_electrodes]
        common_channel_ids.append(list(np.array(chans)[elec_idxs]))

    ### Aggregate recordings ###
    recordings_common = []
    common_channel_list = np.arange(len(common_electrodes))

    for i, rec in enumerate(recordings):
        recording_comm = rec.channel_slice(channel_ids=common_channel_ids[i], renamed_channel_ids=common_channel_list)
        recordings_common.append(recording_comm)

    multirec = si.concatenate_recordings(recordings_common)
    multirec_f = st.bandpass_filter(multirec)

    # set probe
    multirec_f.set_probe(recordings_common[0].get_probe(), in_place=True)

    # remove 10ms at transitions to avoid sorting artifacts
    # TODO fix this
    transitions = []
    end_frame = 0
    for rec in recordings_common:
        end_frame += rec.get_num_samples()
        transitions.append(end_frame)
    multirec_rm = st.remove_artifacts(multirec_f, list_triggers=transitions, ms_before=10, ms_after=10)

    # Spike sort multirec
    print(f"\n\nProcessing multirec of {len(recordings_common)} configurations")

    ### Spike sorting
    if recompute_sorting and output_folder.is_dir():
        shutil.rmtree(output_folder)
    try:
        if not (cache_folder / 'sorting_raw').is_dir():
            print(f'SORTING WITH {sorter}\n')
            t_start_sort = time.time()
            sorting = ss.run_sorter(sorter, multirec_rm, output_folder=output_folder, verbose=True,
                                    **sorter_params)
            sorting = sorting.save(folder=cache_folder / "sorting_raw")
            print(f"\n\nSpike sorting elapsed time {time.time() - t_start_sort} s")
        else:
            print('Skipping since already sorted')
            sorting = si.load_extractor(cache_folder / 'sorting_raw')
    except Exception as e:
        print(f"{sorter} failed on axon scan {axon_scan_folder}:\n{e}")
        continue

    if export_raw_to_phy and not (cache_folder / "phy_raw").is_dir():
        we_raw = si.extract_waveforms(multirec_rm, sorting, folder=cache_folder / "waveforms_raw", load_if_exists=True,
                                      n_jobs=n_jobs, total_memory=total_memory, progress_bar=True)
        print("Exporting raw sorting output to Phy")
        export_to_phy(we_raw, cache_folder / "phy_raw", n_jobs=n_jobs, total_memory=total_memory,
                      progress_bar=True)

    ### Auto-curation ###
    if auto_curate:
        if (cache_folder / 'sorting_curated').is_dir() and not recompute_curation:
            print("Loading cached sorting")
            sorting_curated = si.load_extractor(cache_folder / 'sorting_curated')
        else:
            print('CURATING SORTING OUTPUT\n')
            print(f'N units before curation: {len(sorting.get_unit_ids())}\n')

            print(f"\nExtracting waveforms")
            we_raw = si.extract_waveforms(multirec_rm, sorting, folder=cache_folder / "waveforms_raw",
                                          load_if_exists=True,
                                          n_jobs=n_jobs, total_memory=total_memory, progress_bar=True)

            qc_df = st.compute_quality_metrics(we_raw, metric_names=["firing_rate", "isi_violation", "snr"])

            if fr_thresh is not None:
                print(f"Curation based on firing rate")
                fr_query = f"firing_rate > {fr_thresh}"
                qc_df_selected = qc_df.query(fr_query)
                print(f'N units after num spikes curation: {len(qc_df_selected)}\n')
            else:
                qc_df_selected = qc_df

            if isi_viol_thresh is not None:
                if len(qc_df_selected) > 0:
                    print(f"Curation based on isi violation ratio")
                    isi_query = f"isi_violations_rate < {isi_viol_thresh}"
                    qc_df_selected = qc_df_selected.query(isi_query)
                    print(f'N units after ISI curation: {len(qc_df_selected)}\n')
                else:
                    print("No units remain after curation")

            if snr_thresh is not None:
                if len(qc_df_selected) > 0:
                    print(f"Curation based on SNR")
                    snr_query = f"snr > {snr_thresh}"
                    qc_df_selected = qc_df_selected.query(snr_query)
                    print(f'N units after SNR curation: {len(qc_df_selected)}\n')
                else:
                    print("No units remain after curation")

            keep_unit_ids = qc_df_selected.index.values
            sorting_curated = sorting.select_units(keep_unit_ids)

            coms_raw = st.localize_units(we_raw, num_channels=num_channels_for_com)
            template_metrics_raw = st.calculate_template_metrics(we_raw)

            all_templates = []
            coms = []
            for unit in sorting_curated.get_unit_ids():
                all_templates.append(we_raw.get_template(unit))
                coms.append(coms_raw[unit])
            template_metrics = template_metrics_raw.loc[keep_unit_ids]

            # add COM properties
            sorting_curated.set_property("com", np.array(coms))

            # add QC properties
            for metric_name in qc_df_selected.columns:
                sorting_curated.set_property(metric_name, qc_df_selected[metric_name])

            # add QC properties
            for metric_name in template_metrics.columns:
                sorting_curated.set_property(metric_name, template_metrics[metric_name])

            if export_curated_to_phy:
                print("Exporting auto-curated sorting output to Phy")
                export_to_phy(sorting_curated, cache_folder / "phy_curated", n_jobs=n_jobs, total_memory=total_memory,
                              progress_bar=True)

            if dump_sorting:
                # Dump sorting with templates
                sorting_curated.save(folder=cache_folder / 'sorting_curated')
                print(f"Curated sorting saved to {cache_folder / 'sorting_curated'}\n")
    else:
        print(f"\nExtracting waveforms")
        we_raw = si.extract_waveforms(multirec_rm, sorting, folder=cache_folder / "waveforms_raw",
                                      load_if_exists=True,
                                      n_jobs=n_jobs, total_memory=total_memory, progress_bar=True)

        all_templates = we_raw.get_all_templates()
        sorting.set_property("template", all_templates)
        # compute COM on templates
        coms = st.localize_units(we_raw,
                                               num_channels=num_channels_for_com)

        template_metrics = st.calculate_template_metrics(we_raw)

        # add COM properties
        sorting.set_property("com", np.array(list(coms.values())))

        # add QC properties
        for metric_name in template_metrics.columns:
            sorting.set_property(metric_name, template_metrics[metric_name])

        if dump_sorting:
            # Dump sorting with templates
            sorting.save(folder=cache_folder / 'sorting')
            print(f"Curated sorting saved to {cache_folder / 'sorting'}\n")

    t_stop_proc = time.time()
    print(f"\n\nTotal filtering, sorting, curation elapsed time {np.round(t_stop_proc - t_start_all, 2)} s")

    # ### Split sortings ###
    if not (cache_folder / 'sorting_0').is_dir() or recompute_splitting:
        print("Splitting sortings")
        t_start = time.time()
        multisorting = si.load_extractor(cache_folder / 'sorting_raw')
        curated = False
        if (cache_folder / 'sorting_curated').is_dir():
            multisorting_curated = si.load_extractor(cache_folder / 'sorting_curated')
            curated = True

        # remove zero spikes units
        units_with_spikes = []
        for unit in multisorting.get_unit_ids():
            if len(multisorting.get_unit_spike_train(unit)) > 0:
                units_with_spikes.append(unit)
        multisorting = multisorting.select_units(units_with_spikes)

        if curated:
            units_with_spikes = []
            for unit in multisorting_curated.get_unit_ids():
                if len(multisorting_curated.get_unit_spike_train(unit)) > 0:
                    units_with_spikes.append(unit)
            multisorting_curated = multisorting_curated.select_units(units_with_spikes)

        # get single sortings and remove empty clusters
        sortings = []
        sortings_curated = []
        transitions = []
        end_frame = 0
        for i, rec in enumerate(recordings_common):
            print(f"Splitting sorting {i}")
            start_frame = int(end_frame)
            end_frame += rec.get_num_samples()
            sorting = multisorting.frame_slice(start_frame=start_frame, end_frame=end_frame)
            units_with_spikes = []
            for unit in sorting.get_unit_ids():
                if len(sorting.get_unit_spike_train(unit)) > 0:
                    units_with_spikes.append(unit)
            if len(units_with_spikes) == 0:
                print(f"Skipping sorting {i} because no spikes detected")
                continue
            sorting = sorting.select_units(units_with_spikes)
            if (cache_folder / f'sorting_{i}').is_dir():
                shutil.rmtree(cache_folder / f'sorting_{i}')
            sorting.save(folder=cache_folder / f'sorting_{i}')
            sortings.append(sorting)

            if curated:
                sorting_curated = multisorting_curated.frame_slice(start_frame=start_frame, end_frame=end_frame)
                units_with_spikes = []
                for unit in sorting_curated.get_unit_ids():
                    if len(sorting_curated.get_unit_spike_train(unit)) > 0:
                        units_with_spikes.append(unit)
                if len(units_with_spikes) == 0:
                    print(f"Skipping curated sorting {i} because no spikes detected")
                    continue
                sorting_curated = sorting_curated.select_units(units_with_spikes)
                if (cache_folder / f'sorting_curated_{i}').is_dir():
                    shutil.rmtree(cache_folder / f'sorting_curated_{i}')
                sorting_curated.save(folder=cache_folder / f'sorting_curated_{i}')
                sortings_curated.append(sorting_curated)

        t_stop = time.time()
        print(f"Elapsed time splitting {t_stop - t_start}")
    else:
        print("Sortings already split")

    ### Extract full templates ###
    # load templates and locations
    locations = []
    for i, rec in enumerate(recordings):
        locs = rec.get_channel_locations()
        locations.append(locs)

    # find common locations and merge templates
    locations_set = [set(list([tuple(l) for l in locs])) for locs in locations]

    for l, lset in enumerate(locations_set):
        if l == 0:
            loc_union = lset
            loc_intersect = lset
        else:
            loc_union = loc_union | lset
            loc_intersect = loc_intersect & lset

    locations_all = np.array(list(loc_union))
    print(f"All locations: {len(loc_union)}, Intersection among configurations: {len(loc_intersect)}")

    if (cache_folder / 'sorting_full_templates').is_dir() and not recompute_full_template:
        print("Loading cached sorting with full templates")
        sorting = si.load_extractor(cache_folder / 'sorting_full_templates')
        print(sorting.get_property_keys())
    else:
        print("Extracting full templates")
        t_start = time.time()
        sorting_folders = get_sorting_folders_from_cache(cache_folder)
        sort_ids = [int(sort.stem.split('_')[-1]) for sort in sorting_folders]
        sorting_folders = np.array(sorting_folders)[np.argsort(sort_ids)]

        sortings = [None] * len(recordings)
        units = []
        for i, sort_folder in enumerate(sorting_folders):
            i = sorting_folder_to_index(sort_folder)
            sort = si.load_extractor(sort_folder)
            sortings[i] = sort

        if (cache_folder / 'sorting_curated').is_dir():
            sorting = si.load_extractor(cache_folder / 'sorting_curated')
        else:
            sorting = se.load_extractor(cache_folder / 'sorting')

        # load templates and locations
        rec_filtered = []
        for i, rec in enumerate(recordings):
            rec_filt = st.bandpass_filter(rec)
            rec_filtered.append(rec_filt)

        # Build dictionary with templates and locations for each config
        template_dict = {}
        for u in sorting.get_unit_ids():
            template_dict[u] = {'templates': [],
                                'locations': []}

        template_shape = None
        for i in np.arange(len(rec_filtered)):
            sorting_tmp_folder = cache_folder / f'sorting_templates_{i}'
            recf = rec_filtered[i]
            sort = sortings[i]
            if sort is None:
                print(f"Skipping sorting for recording {i+1} because no units detected")
                continue

            if sorting_tmp_folder.is_dir():
                print("Loading cached sortings with templates")
                sort = si.load_extractor(sorting_tmp_folder)
                temps = sort.get_property('template')
            else:
                print(f"Extracting templates for recording {i + 1} / {len(rec_filtered)}")
                print(f"Total units: {len(sorting.get_unit_ids())} - Units in {i+1}: {len(sort.get_unit_ids())}")
                we_single = si.extract_waveforms(recf, sort, folder=cache_folder / f"waveforms_{i}",
                                                 load_if_exists=True,
                                                 n_jobs=n_jobs, total_memory=total_memory, progress_bar=True)
                temps = we_single.get_all_templates()
                sort.set_property("template", temps)
                sort.save(folder=sorting_tmp_folder)

            if template_shape is None:  # if i == 0:
                template_shape = temps[0].shape
            for (u, temp) in zip(sort.get_unit_ids(), temps):
                template_dict[u]['templates'].append(temp.T)
                template_dict[u]['locations'].append(recf.get_channel_locations())

        # allocate memmap objects for parallel processing
        full_templates = np.memmap(filename=cache_folder / 'full_templates.raw',
                                   mode='w+', dtype='float',
                                   shape=(len(sorting.get_unit_ids()), len(loc_union), template_shape[0]))
        shared_channels = np.memmap(filename=cache_folder / 'shared_channels.raw',
                                    mode='w+', dtype='int',
                                    shape=(len(sorting.get_unit_ids()), len(loc_union)))

        # extract templates in parallel
        Parallel(n_jobs=n_jobs)(
            delayed(extract_one_full_template)(i, template_dict[key], locations_all, full_templates,
                                               shared_channels, True)
            for i, key in enumerate(list(template_dict.keys())))

        # load template as property
        sorting.set_property("template", full_templates)

        sorting.save(folder=cache_folder / 'sorting_full_templates')
        print("Saving locations in npy")
        np.save(cache_folder / "full_locations.npy", locations_all)
        print("Saving templates in npy")
        np.save(cache_folder / "full_templates.npy", full_templates)
        t_stop = time.time()
        print(f"Elapsed time full template extraction: {t_stop - t_start}s")

    if not Path(cache_folder / "full_templates.npy").is_file():
        print("Saving templates in npy")
        full_templates = sorting.get_property("template")
        np.save(cache_folder / "full_templates.npy", full_templates)
    else:
        full_templates = np.load(cache_folder / "full_templates.npy", allow_pickle=True)
    if not Path(cache_folder / "full_locations.npy").is_file():
        print("Saving locations in npy")
        np.save(cache_folder / "full_locations.npy", locations_all)

    print(f"Total units {len(sorting.get_unit_ids())}")

    if plot_unit_templates:
        t_start_plot = time.time()
        figures_units = figures_folder / "units"
        figures_units.mkdir(parents=True, exist_ok=True)

        locations = locations_all
        templates = full_templates
        unit_ids = sorting.get_unit_ids()
        for i_u in tqdm(np.arange(len(unit_ids)), ascii=True,
                        desc=f"Plotting unit templates"):
            u = unit_ids[i_u]
            # com = np.round(np.array(sorting.get_unit_property(u, 'com')), 1)
            template = templates[i_u]

            fig_name = f"unit{u}.pdf"
            fig, axs = plt.subplots(ncols=2, figsize=(10, 6))
            max_chan = np.unravel_index(np.argmax(np.abs(template)), template.shape)[0]
            amp = np.round(np.max(np.abs(template)))

            av.plot_amplitude_map(template, locations, log=True, ax=axs[0])
            axs[0].set_title("Amplitude")
            av.plot_peak_latency_map(template, locations, fs=fs, ax=axs[1])
            axs[1].set_title("Latency")
            fig.suptitle(f"unit {u} - amp {amp}$\mu$V\n")
            fig.savefig(str(figures_units / fig_name))
            plt.close(fig)

        t_stop_plot = time.time()
        print(f"\n\nTotal plotting time {np.round(t_stop_plot - t_start_plot, 2)} s")

    # Compute axon velocities
    t_start_axons = time.time()
    if (cache_folder / "sorting_axons").is_dir() and not recompute_axon_velocities:
        print("Loading cached sorting with axons")
        sorting_axon = si.load_extractor(cache_folder / "sorting_axons")
    else:
        print("Axon tracking params")
        print(params)

        unit_ids = sorting.get_unit_ids()
        templates = full_templates
        locations = locations_all
        units_with_axon = []
        gtrs = []
        for i_u in tqdm(np.arange(len(unit_ids)), ascii=True, desc="Extracting axon velocity"):
            u = unit_ids[i_u]
            template = templates[i_u]
            try:
                gtr = av.compute_graph_propagation_velocity(template, locations, fs, **params)
                if len(gtr.branches) >= 1:
                    gtrs.append(gtr)
                    units_with_axon.append(u)
                    print(f"Found axon unit {u}")
            except Exception as e:
                print(f"Axon tracking failed unit {u}: \nError {e}")

        print(f"Found {len(units_with_axon)} with a detectable axons!")
        if len(units_with_axon) > 0:
            sorting_axon = sorting.select_units(units_with_axon)
            sorting_axon.set_property("gtr", gtrs)

            if dump_sorting and len(units_with_axon) > 0:
                sorting_axon.save(folder=cache_folder / "sorting_axons")
        else:
            sorting_axon = None
    t_stop_axons = time.time()
    print(f"\n\nTotal axon extraction time {np.round(t_stop_axons - t_start_axons, 2)} s")

    if sorting_axon is not None:
        t_start_axon_plot = time.time()
        if plot_unit_axons:
            figures_axons = figures_folder / "units_axons"
            figures_axons.mkdir(parents=True, exist_ok=True)

            gtrs = sorting_axon.get_property("gtr")
            unit_ids = sorting_axon.get_unit_ids()
            for i_u in tqdm(np.arange(len(unit_ids)), ascii=True,
                            desc=f"Plotting unit axons"):
                u = unit_ids[i_u]
                gtr = gtrs[i_u]

                fig_name = f"unit{u}.pdf"
                fig = plt.figure(figsize=(10, 6))
                amp = np.round(np.max(np.abs(gtr.amplitudes)), 2)
                fig, axes = av.plot_axon_summary(gtr, fig=fig)
                fig.suptitle(f"unit {u} - amp {amp}$\mu$V\n")
                fig.savefig(str(figures_axons / fig_name))
                plt.close(fig)
        t_stop_axon_plot = time.time()
        print(f"\n\nTotal axon plotting time {np.round(t_stop_axon_plot - t_start_axon_plot, 2)} s")
    else:
        print("\n\nSkipping plotting axons because no detectable axons\n\n")

    mat_folder_path = Path(mat_folder)
    mat_folder_path.mkdir(exist_ok=True, parents=True)
    mat_files = [file for file in mat_folder_path.iterdir() if file.is_file() and file.suffix == ".mat"]
    if len(mat_files) == len(recordings) and not recompute_mat_files:
        print("Skipping computing .mat files because already computed")
    else:
        print(f"Computing .mat files for {len(recordings)} recordings")
        for rec_i in tqdm(range(len(recordings))):
            recording = recordings[rec_i]
            sorting_folder = cache_folder / f"sorting_templates_{rec_i}"

            locations = recording.get_channel_locations()
            y_max = np.max(locations[:, 1])
            y_min = np.min(locations[:, 1])
            locations[:, 1] = y_max - locations[:, 1] + y_min

            electrodes = recording.get_property("electrode")

            save_dict = {"units": [], "locations": locations, "fs": recording.get_sampling_frequency()}

            if sorting_folder.exists():
                sorting = si.load_extractor(sorting_folder)
                templates = sorting.get_property("template")
                max_channels = np.argmin(np.min(templates, axis=1), axis=1)

                for u_i, u in enumerate(np.atleast_1d(sorting.get_unit_ids())):
                    channel = max_channels[u_i]
                    x_max, y_max = locations[channel]
                    unit_dict = {
                        "spike_train": sorting.get_unit_spike_train(u),
                        "x_max": x_max,
                        "y_max": y_max,
                        "template": templates[u_i, :, :],
                        "unit_id": u,
                        "electrode": electrodes[channel],
                    }
                    save_dict["units"].append(unit_dict)

            parent_folder = mat_folder_path / f"{rec_i}" / r"spikes/raw_data"
            parent_folder.mkdir(parents=True, exist_ok=True)
            savemat(parent_folder / "t_spk_mat_ks.mat", save_dict)
            np.savez(parent_folder / "t_spk_npz_ks.npz",
                     units=save_dict["units"],
                     locations=save_dict["locations"],
                     fs=save_dict["fs"])
