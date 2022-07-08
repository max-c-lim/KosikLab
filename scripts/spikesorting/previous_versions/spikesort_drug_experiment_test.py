"""
Template script to spike sort a list of recordings using SpikeInterface.

Author: Alessio Buccino
"""

import spikeextractors as se  # 0.9.9
import spiketoolkit as st  # 0.7.7
import spikesorters as ss  # 0.4.5
from natsort import natsorted
from pathlib import Path
import time
import numpy as np
import matplotlib as mpl
import os

mpl.use("agg")

############################################
################  INPUTS ###################
############################################

# recordings_path = Path("/home/maxlim/SpikeSorting/data/DL/recordings/maxone_2953.raw.h5")
drug_folder = Path("/home/maxlim/SpikeSorting/data/maxone/200123/2953/recordings/test")
inter_path = Path("/home/maxlim/SpikeSorting/data/dated/220624/alessio_drug_maxone_2953_concat2")
ss.Kilosort2Sorter.set_kilosort2_path("/home/maxlim/SpikeSorting/Kilosort2")
os.environ['HDF5_PLUGIN_PATH'] = '/home/maxlim/SpikeSorting/extra_libraries/'

# sorter name
sorter = 'kilosort2'
# sorter params
sorter_params = {"n_jobs_bin": 16, "chunk_mb": 4000}

############################################
################  PARAMS ###################
############################################

# If True, and spike sorting output is present, it's deleted and resorted
recompute_sorting = True
recompute_curation = True
delete_output_folder = False

dump_recording = False

### FILTER params ###
freq_min = 300
fre_max = 6000

### AUTOMATIC CURATION ###
# If True, output is automatically curated using quality metrics (QC)
auto_curate = True

# Thresholds for automatic curations (if any is None, that autocuration is skipped
# ISI-violation ratio (greater values are removed)
isi_viol_thresh = 0.3
# firing rate (smaller values are removed)
fr_thresh = 0.05
# signal-to-noise ratio (smaller values are removed)
snr_thresh = 5

# TODO: add more QCs

### OTHER PROCESSING PARAMS ###
# number of jobs to use
n_jobs = 16
# number of Mb to use for RAM (4000=4Gb)
chunk_mb = 4000
# Number of spikes per unit to compute templates (None-> all spikes are used)
max_spikes_per_unit = None

########################################################################################################################
# Let's do it!

RECORDING_FILES = [
    Path("/home/maxlim/SpikeSorting/data/DL/recordings/5116.raw.h5"),
]

INTERMEDIATE_FOLDERS = [
    Path("/home/maxlim/SpikeSorting/data/dated/220707/al_5116"),
]

for drug_folder, inter_path in zip(RECORDING_FILES, INTERMEDIATE_FOLDERS):
    print(f"Processing: {drug_folder}")

    inter_path.mkdir(parents=True, exist_ok=True)
    t_start_all = time.time()

    output_folder = inter_path / 'sorted' / sorter
    cache_folder = inter_path / 'cache'
    cache_folder.mkdir(parents=True, exist_ok=True)
    sorting_cache = cache_folder / 'sorted' / sorter
    sorting_cache.mkdir(parents=True, exist_ok=True)
    # figures_folder = inter_path / 'figures'
    # figures_folder.mkdir(parents=True, exist_ok=True)
    tmp_folder = cache_folder / 'tmp' / sorter
    tmp_folder.mkdir(parents=True, exist_ok=True)

    if str(drug_folder).endswith(".raw.h5"):
        recording = se.MaxOneRecordingExtractor(drug_folder, load_spikes=False)
    else:
        # CONCATENATING RECORDINGS
        # retrieve recordings
        recordings = []
        epoch_names = []

        experiment_names = natsorted([p.name for p in drug_folder.iterdir() if "h5" in p.suffix])

        for exp in experiment_names:
            mea_file = drug_folder / exp
            rec = se.MaxOneRecordingExtractor(mea_file, load_spikes=False)
            recordings.append(rec)
            epoch_names.append(exp)
            print(f"DURATION {exp}: {rec.get_num_frames() / rec.get_sampling_frequency()} s -- "
                  f"NUM. CHANNELS: {rec.get_num_channels()}")

        print(f"Concatenating {len(recordings)} recordings")
        print(f"Epochs names: {epoch_names}")
        recording = se.MultiRecordingTimeExtractor(recordings, epoch_names=epoch_names)


    if (cache_folder / 'recording.pkl').is_file() and (cache_folder / 'recording.dat').is_file():
        print("Loading cached recording")
        rec_cache = se.load_extractor_from_pickle(cache_folder / 'recording.pkl')
        fs = rec_cache.get_sampling_frequency()
    else:
        print('FILTERING\n')
        start = time.time()
        rec_f = st.preprocessing.bandpass_filter(recording, freq_min=freq_min, freq_max=fre_max, dtype='float32', filter_type="butter")
        rec_cache = se.CacheRecordingExtractor(rec_f, save_path=cache_folder / 'recording.dat',
                                               chunk_mb=chunk_mb, n_jobs=n_jobs, verbose=True)
        stop = time.time()
        print(f'Elapsed filter time {np.round(stop - start, 2)}\n')
        fs = rec_f.get_sampling_frequency()

        if dump_recording:
            # Dump recording with activity
            rec_cache.dump_to_pickle(cache_folder / 'recording.pkl')
            print(f"Filtered recording dumped to {cache_folder / 'recording.pkl'}\n")

    ### Spike sorting
    if recompute_sorting and (sorting_cache / 'sorting.npz').is_file():
        (sorting_cache / 'sorting.npz').unlink()
    try:
        if not (sorting_cache / 'sorting.npz').is_file():
            print(f'SORTING WITH {sorter}\n')
            t_start_sort = time.time()
            sorting = ss.run_sorter(sorter, rec_cache, output_folder=output_folder, verbose=True,
                                    delete_output_folder=False, **sorter_params)
            sorting = se.CacheSortingExtractor(sorting, save_path=str(sorting_cache / 'sorting.npz'))
            sorting.dump_to_pickle(sorting_cache / 'sorting.pkl')
            print(f"\n\nSpike sorting elapsed time {time.time() - t_start_sort} s")
        else:
            print(f'Skipping sorting since already sorted')
            sorting = se.load_extractor_from_pickle(sorting_cache / 'sorting.pkl')
    except Exception as e:
        print(f"{sorter} failed on recording \n\nError: {e}")
        continue

    print(f"Total units found by {sorter}: {len(sorting.get_unit_ids())}")

    # if export_raw_to_phy and not (sorting_cache / "phy_raw").is_folder():
    #     print("Exporting raw sorting output to Phy")
    #     st.postprocessing.export_to_phy(rec_cache, sorting, cache_folder / "phy_raw",
    #                                     n_jobs=n_jobs, chunk_mb=chunk_mb)

    ### Auto-curation ###
    if auto_curate:
        if (sorting_cache / 'sorting_curated.pkl').is_file() and not recompute_curation:
            print("Loading cached sorting")
            sorting_curated = se.load_extractor_from_pickle(sorting_cache / 'sorting_curated.pkl')
        else:
            print('CURATING\n')
            print(f'N units before curation: {len(sorting.get_unit_ids())}\n')
            if fr_thresh is not None:
                print(f"Curation based on firing rate")
                sorting_curated = st.curation.threshold_firing_rates(sorting, threshold=fr_thresh,
                                                                     threshold_sign='less',
                                                                     duration_in_frames=rec_cache.get_num_frames())
                print(f'N units after num spikes curation: {len(sorting_curated.get_unit_ids())}\n')
            else:
                sorting_curated = sorting

            if fr_thresh is not None:
                if len(sorting_curated.get_unit_ids()) > 0:
                    print(f"Curation based on isi violation ratio")
                    sorting_curated = st.curation.threshold_isi_violations(sorting_curated,
                                                                           threshold=isi_viol_thresh,
                                                                           threshold_sign='greater',
                                                                           duration_in_frames=
                                                                           rec_cache.get_num_frames())
                    print(f'N units after ISI curation: {len(sorting_curated.get_unit_ids())}\n')
                else:
                    print("No units remain after curation")

            if snr_thresh is not None:
                if len(sorting_curated.get_unit_ids()) > 0:
                    print(f"Curation based on SNR (this might take some time...)")
                    # Compute waveforms and templates
                    sorting_curated.set_tmp_folder(tmp_folder)
                    print(f"\t\tExtracting waveforms")
                    wf = st.postprocessing.get_unit_waveforms(rec_cache, sorting_curated, chunk_mb=chunk_mb,
                                                              max_spikes_per_unit=max_spikes_per_unit,
                                                              n_jobs=n_jobs, verbose=True)
                    print("Done computing waveforms")
                    print(f"\t\tExtracting templates")
                    templates_curated = st.postprocessing.get_unit_templates(rec_cache, sorting_curated)
                    print("Done computing templates")
                    sorting_curated.clear_units_spike_features("waveforms")
                    sorting_curated = st.curation.threshold_snrs(sorting_curated, rec_cache,
                                                                 threshold=snr_thresh,
                                                                 threshold_sign='less')
                    print(f'N units after SNR curation: {len(sorting_curated.get_unit_ids())}\n')
                else:
                    print("No units remain after curation")

            # # compute COM and features
            # coms = st.postprocessing.compute_unit_centers_of_mass(rec_cache, sorting_curated,
            #                                                       num_channels=num_channels_for_com)
            #
            # # feats = st.postprocessing.compute_unit_template_features(rec_cache, sorting_curated)
            #
            if True:  # export_curated_to_phy
                print("Exporting auto-curated sorting output to Phy")
                st.postprocessing.export_to_phy(rec_cache, sorting_curated, sorting_cache / "phy_curated",
                                                n_jobs=n_jobs, chunk_mb=chunk_mb)
            #
            # if dump_sorting:
            #     # Dump sorting with templates
            #     sorting_curated.dump_to_pickle(sorting_cache / 'sorting_curated.pkl', include_features=False)
            #     print(f"Curated sorting dumped to {sorting_cache / 'sorting_curated.pkl'}\n")
            #     shutil.rmtree(tmp_folder, ignore_errors=True)
    # else:
    #     sorting.set_tmp_folder(tmp_folder)
    #     # compute templates anyways
    #     wf = st.postprocessing.get_unit_waveforms(rec_cache, sorting, chunk_mb=chunk_mb,
    #                                               max_spikes_per_unit=max_spikes_per_unit,
    #                                               n_jobs=n_jobs, verbose=True)
    #     templates = st.postprocessing.get_unit_templates(rec_cache, sorting)
    #
    #     # compute COM on templates
    #     coms = st.postprocessing.compute_unit_centers_of_mass(rec_cache, sorting,
    #                                                           num_channels=num_channels_for_com)
    #
    #     # feats = st.postprocessing.compute_unit_template_features(rec_cache, sorting)
    #
    #     if dump_sorting:
    #         # Dump sorting with templates
    #         sorting.dump_to_pickle(sorting_cache / 'sorting_uncurated.pkl', include_features=False)
    #         print(f"Curated sorting dumped to {sorting_cache / 'sorting_uncurated.pkl'}\n")
    #         shutil.rmtree(tmp_folder, ignore_errors=True)

    # if auto_curate:
    #     sorting_output = sorting_curated
    # else:
    #     sorting_output = sorting

    # get single sortings and remove empty clusters
    # print("Splitting sorting output in epochs")
    # sortings = []
    # epoch_names = recording.get_epoch_names()
    # print("Epochs:", epoch_names)
    # for i, ep_name in enumerate(epoch_names):
    #     info = recording.get_epoch_info(ep_name)
    #
    #     start_frame = info['start_frame']
    #     end_frame = info['end_frame']
    #
    #     subrec = se.SubRecordingExtractor(rec_cache, start_frame=start_frame, end_frame=end_frame)
    #     subsort = se.SubSortingExtractor(sorting_output, start_frame=start_frame, end_frame=end_frame)
    #     subsort = st.curation.threshold_num_spikes(subsort, 1, 'less')  # remove units with no spikes in epoch
    #
    #     # save output
    #     assert "template" in subsort.get_shared_unit_property_names(), "No template found in properties"
    #
    #     subsort.dump_to_pickle(sorting_cache / f'sorting_{ep_name}.pkl', include_properties=True,
    #                            include_features=False)

    t_stop_proc = time.time()
    print(f"\n\nTotal filtering, sorting, curation elapsed time {np.round(t_stop_proc - t_start_all, 2)} s")
    t_start_plot = time.time()

    # if plot_unit_templates:
    #     print(f"Plotting centers of mass figure")
    #     # figure with COM
    #     w_elec = sw.plot_electrode_geometry(rec_cache, color='y')
    #     cmap = 'coolwarm'
    #     cm = plt.get_cmap(cmap)
    #     n_units = len(sorting_output.get_unit_ids())
    #
    #     # precompute template amplitudes
    #     template_amps = []
    #     for i_u, u in enumerate(sorting_output.get_unit_ids()):
    #         template = sorting_output.get_unit_property(u, 'template')
    #         amp = np.max(np.abs(template))
    #         template_amps.append(amp)
    #
    #     norm_amps = np.array(template_amps) - np.min(template_amps)
    #     norm_amps /= np.ptp(template_amps)
    #
    #     for i_u, u in enumerate(sorting_output.get_unit_ids()):
    #         com = sorting_output.get_unit_property(u, 'com')
    #         color = cm(norm_amps[i_u])
    #         w_elec.ax.plot(com[0], com[1], marker='o', color=color, markersize=5)
    #
    #     colorbar = add_colorbar(rec_cache.get_channel_locations(), template_amps, w_elec, cmap)
    #     colorbar.set_label("Amplitude $\mu$V", rotation=90, labelpad=-20)
    #
    #     x, y = rec_cache.get_channel_locations().T
    #     w_elec.ax.text(np.min(x) - 20, np.min(y) - 20, "0", fontsize=10)
    #     w_elec.figure.savefig(figures_folder / 'com.png', dpi=600)
    #
    #     figures_units = figures_folder / "units"
    #     figures_units.mkdir(parents=True, exist_ok=True)
    #
    #     locations = rec_cache.get_channel_locations()
    #     fs = rec_cache.get_sampling_frequency()
    #     electrode_ids = np.array([rec_cache.get_channel_property(ch, 'electrode')
    #                               for ch in rec_cache.get_channel_ids()])
    #     unit_ids = sorting_output.get_unit_ids()
    #     for i_u in tqdm(np.arange(len(unit_ids)), ascii=True,
    #                     desc=f"Plotting unit templates"):
    #         u = unit_ids[i_u]
    #         com = np.round(np.array(sorting_output.get_unit_property(u, 'com')), 1)
    #         template = sorting_output.get_unit_property(u, 'template')
    #
    #         fig_name = f"unit{u}.pdf"
    #         fig, axs = plt.subplots(ncols=2, figsize=(10, 6))
    #         max_chan = np.unravel_index(np.argmax(np.abs(template)), template.shape)[0]
    #         electrode_id = electrode_ids[max_chan]
    #         amp = np.round(np.max(np.abs(template)))
    #
    #         av.plot_amplitude_map(template, locations, log=True, ax=axs[0])
    #         axs[0].set_title("Amplitude")
    #         av.plot_peak_latency_map(template, locations, ax=axs[1])
    #         axs[1].set_title("Latency")
    #         fig.suptitle(f"unit {u} - elec {electrode_id} - amp {amp}$\mu$V\n"
    #                      f"com: {com}")
    #         fig.savefig(str(figures_units / fig_name))
    #         plt.close(fig)
    # t_stop_plot = time.time()
    # print(f"\n\nTotal plotting time {np.round(t_stop_plot - t_start_plot, 2)} s")

    t_stop_all = time.time()
    #print(f"\n\nTotal elapsed time to analyze {recording_path} \n{np.round(t_stop_all - t_start_all, 2)} s")
