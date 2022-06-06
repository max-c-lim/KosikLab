"""
Template script to spike sort a list of recordings using SpikeInterface>=0.90.

Author: Alessio Buccino
"""

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw
from spikeinterface.exporters import export_to_phy

import shutil
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

mpl.use("agg")

# from spikesorting_utils import add_colorbar
# import axon_velocity as av

############################################
################  INPUTS ###################
############################################


# list of data files you want to spike sort
recording_files = [
    "/home/maxlim/SpikeSorting/data/maxone/200123/2950/recordings/0uM.raw.h5"
]
# List of intermediate folders where tmp and output files are saved
intermediate_folders = [
    "/home/maxlim/SpikeSorting/data/maxone/si/2950_si",
]
assert len(recording_files) == len(intermediate_folders), "'recording_files' and 'intermediate_folders' " \
                                                          "should have the same length"

# sorter name
sorter = 'kilosort2'
# sorter params
sorter_params = {"n_jobs_bin": 64, "TOTAL_MEMORY": "4G"}

############################################
################  PARAMS ###################r 
############################################

# If True, and spike sorting output is present, it's deleted and resorted
recompute_sorting = False
recompute_curation = True

# If True, filtered data and sorted outputs are saved in a format that it's easy to retrieve (.pkl)
dump_recording = False
dump_sorting = False

# If True, exports to Phy
export_raw_to_phy = False
export_curated_to_phy = False

# If True, unit templates are plotted for all units
plot_unit_templates = False
plot_image = False

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


### OTHER PROCESSING PARAMS ###
# number of jobs to use
n_jobs = 64
# total RAM to use
total_memory = "4G"
# Number of spikes per unit to compute templates (None-> all spikes are used)
max_spikes_per_unit = None
# Number of channels to compute center of mass
num_channels_for_com = 30

# set toolbox paths
ss.Kilosort2Sorter.set_kilosort2_path("/home/maxlim/SpikeSorting/Kilosort2")
os.environ['HDF5_PLUGIN_PATH'] = '/home/maxlim/SpikeSorting/extra_libraries/'
		
########################################################################################################################
# Let's do it!

for (rec_path, inter_path) in zip(recording_files, intermediate_folders):
    t_start_all = time.time()
    # set paths
    rec_path = Path(rec_path)
    inter_path = Path(inter_path)

    print(f"Processing recording: {rec_path}")

    output_folder = inter_path / 'sorted' / sorter
    cache_folder = inter_path / 'cache'
    figures_folder = inter_path / 'figures'
    cache_folder.mkdir(parents=True, exist_ok=True)
    figures_folder.mkdir(parents=True, exist_ok=True)
    tmp_folder = cache_folder / 'tmp' / sorter 
    tmp_folder.mkdir(parents=True, exist_ok=True)

    # Load recording
    try:
        print("Trying to load Maxwell recording")
        rec = se.MaxwellRecordingExtractor(rec_path)
    except:
        raise Exception(f"Could not open the provided file: {rec_path} with the MaxwellRecordingExtractor")

    print(f"DURATION: {rec.get_num_frames() / rec.get_sampling_frequency()} s -- "
          f"NUM. CHANNELS: {rec.get_num_channels()}")

    ### Filter and dumping
    if (cache_folder / 'recording').is_dir():
        print("Loading saved recording")
        rec_cache = si.load_extractor(cache_folder / 'recording')
    else:
        print('FILTERING\n')
        rec_f = st.preprocessing.bandpass_filter(rec, freq_min=freq_min, freq_max=fre_max)

        if dump_recording:
            start = time.time()
            rec_cache = rec_f.save(folder=cache_folder / "recording", n_jobs=n_jobs, total_memory=total_memory,
                                   progress_bar=True)
            stop = time.time()
            print(f'Elapsed saving time {np.round(stop - start, 2)}\n')
            print(f"Filtered recording saved to {cache_folder / 'recording'}\n")
        else:
            rec_cache = rec_f
    fs = rec_cache.get_sampling_frequency()

    ### Spike sorting
    if recompute_sorting and output_folder.is_dir():
        shutil.rmtree(output_folder)
    try:
        if False:  # not (cache_folder / 'sorting_raw').is_dir():
            print(f'SORTING WITH {sorter}\n')
            t_start_sort = time.time()

            sorting = ss.run_sorter(sorter, rec_cache, raise_error=False, output_folder=output_folder, verbose=True,
                                    **sorter_params)
            sorting = sorting.save(folder=cache_folder / "sorting_raw")
            print(f"\n\nSpike sorting elapsed time {time.time() - t_start_sort} s")
        else:
            print('Skipping', rec_path, ' since already sorted')
            from spikeinterface.sorters.kilosort2 import kilosort2
            sorting = kilosort2.Kilosort2Sorter._get_result_from_folder(output_folder)
            sorting = sorting.save(folder=cache_folder / "sorting_raw")
            # sorting = si.load_extractor(cache_folder / 'sorting_raw')
    except Exception as e:
        print(f"{sorter} failed on recording {rec_path}\n\nError: {e}")
        continue

    if export_raw_to_phy and not (cache_folder / "phy_raw").is_dir():
        we_raw = si.extract_waveforms(rec_cache, sorting, folder=cache_folder / "waveforms_raw", load_if_exists=True,
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
            we_raw = si.extract_waveforms(rec_cache, sorting, folder=cache_folder / "waveforms_raw",
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
                
            # # # do same for curated
            we_curated = we_raw.select_units(unit_ids=keep_unit_ids, new_folder=cache_folder / "waveforms_curated")

            coms = st.compute_center_of_mass(we_curated, num_channels=num_channels_for_com)
            template_metrics = st.calculate_template_metrics(we_curated)
            all_templates = we_curated.get_all_templates()
            
            sorting_curated = we_curated.sorting

            # add COM properties
            sorting_curated.set_property("com", coms)

            # add QC properties
            for metric_name in qc_df_selected.columns:
                sorting_curated.set_property(metric_name, qc_df_selected[metric_name])

            # add QC properties
            for metric_name in template_metrics.columns:
                sorting_curated.set_property(metric_name, template_metrics[metric_name])    
                
            # # # # # # # # #

            if export_curated_to_phy:
                print("Exporting auto-curated sorting output to Phy")
                we_curated = si.extract_waveforms(rec_cache, sorting_curated, folder=cache_folder / "waveforms_curated",
                                              load_if_exists=True,
                                              n_jobs=n_jobs, total_memory=total_memory, progress_bar=True)

                export_to_phy(we_curated, cache_folder / "phy_curated", n_jobs=n_jobs, total_memory=total_memory,
                              progress_bar=True)

            if dump_sorting:
                # Dump sorting with templates
                sorting_curated.save(folder=cache_folder / 'sorting_curated')
                print(f"Curated sorting saved to {cache_folder / 'sorting_curated'}\n")
    else:
        print(f"\nExtracting waveforms")
        we_raw = si.extract_waveforms(rec_cache, sorting, folder=cache_folder / "waveforms_raw",
                                      load_if_exists=True,
                                      n_jobs=n_jobs, total_memory=total_memory, progress_bar=True)

        all_templates = we_raw.get_all_templates()
        sorting.set_property("template", all_templates)
        # compute COM on templates
        coms = st.compute_center_of_mass(we_raw, num_channels=num_channels_for_com)

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

 
