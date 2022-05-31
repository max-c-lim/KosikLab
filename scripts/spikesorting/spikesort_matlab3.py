######################################################
################  FOLDER LOCATIONS ###################
######################################################
# List of data files (.raw.h5) you want to spike sort
# If path is a folder with multiple recording files,
# they will be concatenated in natural ordering.
recording_files = [
    # "/home/maxlim/SpikeSorting/data/DL/recordings/maxone_2950.raw.h5",
    "/home/maxlim/SpikeSorting/data/DL/recordings/maxone_2953.raw.h5",
    # "/home/maxlim/SpikeSorting/data/DL/recordings/maxone_2954.raw.h5",
    # "/home/maxlim/SpikeSorting/data/DL/recordings/maxone_2957.raw.h5",
    # "/home/maxlim/SpikeSorting/data/DL/recordings/maxone_5116.raw.h5",
    # "/home/maxlim/SpikeSorting/data/DL/recordings/maxone_5118.raw.h5"
]
# List of intermediate folders where Kilosort2 files and waveforms are saved
intermediate_folders = [
    # "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_2950",
    "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_2953",
    # "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_2954",
    # "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_2957",
    # "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_5116",
    # "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_5118"
]

# List of output folders where final matlab files are saved.
# Matlab files will have the same name as recording files but will end with _sorted.mat
matlab_folders = [
    # "/home/maxlim/SpikeSorting/data/DL/sorted",
    "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_2953",
    # "/home/maxlim/SpikeSorting/data/DL/sorted",
    # "/home/maxlim/SpikeSorting/data/DL/sorted",
    # "/home/maxlim/SpikeSorting/data/DL/sorted"
]

assert len(recording_files) == len(intermediate_folders) == len(matlab_folders), "'recording_files'" \
                                                                                 " 'intermediate_folders' " \
                                                                                 "and 'matlab_folders' " \
                                                                                 "should all have the same length"

######################################################
###############  KILOSORT PARAMETERS  ################
######################################################
# Set toolbox paths
kilosort_path = "/home/maxlim/SpikeSorting/Kilosort2"  # "/home/vandermolen/Kilosort"
hdf5_plugin_path = '/home/maxlim/SpikeSorting/extra_libraries/'  # '/home/vandermolen/SpikeSorting/extra_libraries/'

# Kilosort2 params
kilosort_params = {
    'detect_threshold': 6,
    'projection_threshold': [10, 4],
    'preclust_threshold': 8,
    'car': True,
    'minFR': 0.1,
    'minfr_goodchannels': 0.1,
    'freq_min': 150,
    'sigmaMask': 30,
    'nPCs': 3,
    'ntbuff': 64,
    'nfilt_factor': 4,
    'NT': None,
    'keep_good_only': False,
}

######################################################
###############  RECOMPUTE RESULTS?  #################
######################################################
# If True and exists, its entire parent folder is deleted and recomputed
# (5/19/2022) If False and exists and new recording is specified in recording_files, it will not be computed and the old data will be used.
recompute_recording = False  # Refers to the .dat recording file created for Kilosort2. If True, the 3 other recompute variables become True too
recompute_sorting = False  # If True, entire intermediate_folder will be deleted
recompute_waveforms = False
recompute_curation = True

# Override matlab file if it exists
override_matlab = True

######################################################
##################  COPY SCRIPT?  ####################
######################################################
# If True, a copy of the script will be saved to each path in 'intermediate_folders' with name prepended with "year_month_day_hourminute"
save_script = True

######################################################
##########  PARALLEL PROCESSING PARAMETERS  ##########
######################################################
# n_jobs and total_memory affects how bandpass filter is applied to raw recording, causing different .dat file
# Number of jobs to use for converting raw recording, extracting waveforms, and curation
n_jobs = 16
# Total RAM to use for converting raw extracting waveforms and curation
total_memory = "10G"
# RAM for each chunk for applying bandpass filter and converting .raw.h5 to .dat
chunk_memory = "4G"

######################################################
############  BANDPASS FILTER PARAMETERS  ############
######################################################
freq_min = 300
freq_max = 6000

######################################################
###############  WAVEFORM PARAMETERS  ################
######################################################
# ms before waveform trough to include
ms_before = 5.
# ms after waveform trough to include
ms_after = 5.
# If True and recording has gain_to_uV/offset_to_uV properties, waveforms are converted to uV
return_scaled = True
# Maximum number of waveforms (spikes) per unit to compute templates and extract (None-> all spikes are used)
max_waveforms_per_unit = None
# Data type of waveforms (None-> the recording's datatype is maintained)
waveform_dtype = None
# NOTES:
# Waveforms are centered by negative peak
# Waveforms in .npy files have shape (num_spikes, num_samples, num_channels)
# num_samples = nbefore + 1 (1 for location of spike (spike time) given by Kilosort2) + nafter-1

######################################################
###############  CURATION PARAMETERS  ################
######################################################
# If True, units are automatically curated using quality metrics
# Units that do not meet all thresholds are removed.
auto_curate = True

# If any is None, that auto-curation is skipped.

# Minimum number of spikes per unit
# (smaller values are removed)
min_spikes_per_unit = 50

# Firing rate
# (smaller values are removed)
fr_thresh = 0.05

# ISI-violation ratio (ratio of violation_rate / total_spike_rate)
# (greater values are removed)
isi_viol_thresh = 0.3

# Maximum normalized standard deviation (standard deviation of waveform divided (normalized) by amplitude)
# (greater values are removed)
max_norm_std = 0.3
# If True, use the standard deviation at the peak. If False, use the average standard deviation over the waveform window
max_norm_at_peak = True
# If max_norm_at_peak = False, the waveform window for the average standard deviation
max_norm_over_window_ms_before = 0.5
max_norm_over_window_ms_after = 1.5

# Signal-to-noise ratio (smaller values are removed)
snr_thresh = 5

######################################################
################  MATLAB PARAMETERS  #################
######################################################
# Parameters for resulting .mat file that contains sorted output
# If True, include the electrode number [1, 26400] to the "unit" data structure in the sorted .mat file
save_electrodes = True
# If True, Kilosort2's outputted spike times will be adjusted to align with the negative peak of the spike
# If False, Kilosort2's outputted spike times will be retained, meaning some spike times will not be on the peak of the spike
# If the peak lasts for multiple timepoints, the middle timepoint will be used (num_timepoints//2 = peak_location)
# NOTE: The unit templates will still be centered using Kilosort2's outputted spike times, NOT on the negative peak
# Therefore, if True, the spike times in "spike_train" may not align exactly with "template"
# Will be set to False is "max_waveforms_per_unit" is not None
center_spike_times_on_peak = True
# If True and "center_spike_times_on_peak" is also True, the offsets between Kilosort2's outputted spike times
# and the adjusted spike times on the negative peak will be saved in "spike_train_offsets"
save_spike_train_offsets = center_spike_times_on_peak

######################################################
######################  CODE  ########################
######################################################
# region Setup
def print_stage(text):
    text = str(text)

    num_chars = 60
    char = "="
    indent = int((num_chars - len(text)) / 2)

    print()
    print(num_chars * char)
    print(indent * " " + text)
    print(num_chars * char)


# region Import Modules
print_stage("IMPORTING LIBRARIES")
import time
_import_start = time.time()

from spikeinterface.extractors import MaxwellRecordingExtractor
from spikeinterface.toolkit.preprocessing import bandpass_filter
import numpy as np

import spikeinterface.core.segmentutils as si_segmentutils
from spikeinterface.extractors import BinaryRecordingExtractor

import subprocess
import tempfile
import shutil
import signal
from pathlib import Path
from typing import Optional, List, Any, Union
from tqdm import tqdm
from scipy.io import savemat
import pandas as pd
from natsort import natsorted
print(f"Done. Time: {time.time()-_import_start:.2f}s")
# endregion


# region Kilosort
class RunKilosort:
    # Must create Kilosort instance to 1) set paths 2) check if kilosort is installed
    def __init__(self):
        # Set paths
        self.path = self.set_kilosort_path(kilosort_path)
        os.environ['HDF5_PLUGIN_PATH'] = hdf5_plugin_path

        # Check if kilosort is installed
        if not self.check_if_installed():
            raise Exception(f"Kilosort2 is not installed.")

        # Make sure parameters are formatted correctly
        RunKilosort.format_params()

    # Run kilosort
    def run(self, recording, recording_dat_path, output_folder):
        # STEP 1) Creates kilosort and recording files needed to run kilosort
        self.setup_recording_files(recording, recording_dat_path, output_folder)

        # STEP 2) Actually run kilosort
        self.start_sorting(output_folder, raise_error=True, verbose=True)

        # STEP 3) Return results of Kilosort as Python object for auto curation
        return RunKilosort.get_result_from_folder(output_folder)

    def setup_recording_files(self, recording, recording_dat_path, output_folder):
        # Prepare electrode positions for this group (only one group, the split is done in spikeinterface's basesorter)
        groups = [1] * recording.get_num_channels()
        positions = np.array(recording.get_channel_locations())
        if positions.shape[1] != 2:
            raise RuntimeError("3D 'location' are not supported. Set 2D locations instead")

        # region Make substitutions in txt files to set kilosort parameters
        # region Config text
        kilosort2_master_txt = '''try
            % prepare for kilosort execution
            addpath(genpath('{kilosort2_path}'));

            % set file path
            fpath = '{output_folder}';

            % add npy-matlab functions (copied in the output folder)
            addpath(genpath(fpath));

            % create channel map file
            run(fullfile('{channel_path}'));

            % Run the configuration file, it builds the structure of options (ops)
            run(fullfile('{config_path}'))

            ops.trange = [0 Inf]; % time range to sort

            % preprocess data to create temp_wh.dat
            rez = preprocessDataSub(ops);

            % time-reordering as a function of drift
            rez = clusterSingleBatches(rez);

            % main tracking and template matching algorithm
            rez = learnAndSolve8b(rez);

            % final merges
            rez = find_merges(rez, 1);

            % final splits by SVD
            rez = splitAllClusters(rez, 1);

            % final splits by amplitudes
            rez = splitAllClusters(rez, 0);

            % decide on cutoff
            rez = set_cutoff(rez);

            fprintf('found %d good units \\n', sum(rez.good>0))

            fprintf('Saving results to Phy  \\n')
            rezToPhy(rez, fullfile(fpath));
        catch
            fprintf('----------------------------------------');
            fprintf(lasterr());
            settings  % https://www.mathworks.com/matlabcentral/answers/1566246-got-error-using-exit-in-nodesktop-mode
            quit(1);
        end
        settings  % https://www.mathworks.com/matlabcentral/answers/1566246-got-error-using-exit-in-nodesktop-mode
        quit(0);'''
        kilosort2_config_txt = '''ops.NchanTOT            = {nchan};           % total number of channels (omit if already in chanMap file)
        ops.Nchan               = {nchan};           % number of active channels (omit if already in chanMap file)
        ops.fs                  = {sample_rate};     % sampling rate

        ops.datatype            = 'dat';  % binary ('dat', 'bin') or 'openEphys'
        ops.fbinary             = fullfile('{dat_file}'); % will be created for 'openEphys'
        ops.fproc               = fullfile(fpath, 'temp_wh.dat'); % residual from RAM of preprocessed data
        ops.root                = fpath; % 'openEphys' only: where raw files are
        % define the channel map as a filename (string) or simply an array
        ops.chanMap             = fullfile('chanMap.mat'); % make this file using createChannelMapFile.m

        % frequency for high pass filtering (150)
        ops.fshigh = {freq_min};

        % minimum firing rate on a "good" channel (0 to skip)
        ops.minfr_goodchannels = {minfr_goodchannels};

        % threshold on projections (like in Kilosort1, can be different for last pass like [10 4])
        ops.Th = {projection_threshold};

        % how important is the amplitude penalty (like in Kilosort1, 0 means not used, 10 is average, 50 is a lot)
        ops.lam = 10;

        % splitting a cluster at the end requires at least this much isolation for each sub-cluster (max = 1)
        ops.AUCsplit = 0.9;

        % minimum spike rate (Hz), if a cluster falls below this for too long it gets removed
        ops.minFR = {minFR};

        % number of samples to average over (annealed from first to second value)
        ops.momentum = [20 400];

        % spatial constant in um for computing residual variance of spike
        ops.sigmaMask = {sigmaMask};

        % threshold crossings for pre-clustering (in PCA projection space)
        ops.ThPre = {preclust_threshold};
        %% danger, changing these settings can lead to fatal errors
        % options for determining PCs
        ops.spkTh           = -{kilo_thresh};      % spike threshold in standard deviations (-6)
        ops.reorder         = 1;       % whether to reorder batches for drift correction.
        ops.nskip           = 25;  % how many batches to skip for determining spike PCs

        ops.CAR             = {use_car}; % perform CAR

        ops.GPU                 = 1; % has to be 1, no CPU version yet, sorry
        % ops.Nfilt             = 1024; % max number of clusters
        ops.nfilt_factor        = {nfilt_factor}; % max number of clusters per good channel (even temporary ones) 4
        ops.ntbuff              = {ntbuff};    % samples of symmetrical buffer for whitening and spike detection 64
        ops.NT                  = {NT}; % must be multiple of 32 + ntbuff. This is the batch size (try decreasing if out of memory).  64*1024 + ops.ntbuff
        ops.whiteningRange      = 32; % number of channels to use for whitening each channel
        ops.nSkipCov            = 25; % compute whitening matrix from every N-th batch
        ops.scaleproc           = 200;   % int16 scaling of whitened data
        ops.nPCs                = {nPCs}; % how many PCs to project the spikes into
        ops.useRAM              = 0; % not yet available

        %%'''
        kilosort2_channelmap_txt = '''%  create a channel map file

        Nchannels = {nchan}; % number of channels
        connected = true(Nchannels, 1);
        chanMap   = 1:Nchannels;
        chanMap0ind = chanMap - 1;

        xcoords = {xcoords};
        ycoords = {ycoords};
        kcoords   = {kcoords};

        fs = {sample_rate}; % sampling frequency
        save(fullfile('chanMap.mat'), ...
            'chanMap','connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs')'''
        # endregion
        kilosort2_master_txt = kilosort2_master_txt.format(
            kilosort2_path=str(Path(self.path).absolute()),
            output_folder=str(output_folder.absolute()),
            channel_path=str((output_folder / 'kilosort2_channelmap.m').absolute()),
            config_path=str((output_folder / 'kilosort2_config.m').absolute()),
        )

        kilosort2_config_txt = kilosort2_config_txt.format(
            nchan=recording.get_num_channels(),
            sample_rate=recording.get_sampling_frequency(),
            dat_file=str(recording_dat_path.absolute()),
            projection_threshold=kilosort_params['projection_threshold'],
            preclust_threshold=kilosort_params['preclust_threshold'],
            minfr_goodchannels=kilosort_params['minfr_goodchannels'],
            minFR=kilosort_params['minFR'],
            freq_min=kilosort_params['freq_min'],
            sigmaMask=kilosort_params['sigmaMask'],
            kilo_thresh=kilosort_params['detect_threshold'],
            use_car=kilosort_params['car'],
            nPCs=int(kilosort_params['nPCs']),
            ntbuff=int(kilosort_params['ntbuff']),
            nfilt_factor=int(kilosort_params['nfilt_factor']),
            NT=int(kilosort_params['NT'])
        )

        kilosort2_channelmap_txt = kilosort2_channelmap_txt.format(
            nchan=recording.get_num_channels(),
            sample_rate=recording.get_sampling_frequency(),
            xcoords=[p[0] for p in positions],
            ycoords=[p[1] for p in positions],
            kcoords=groups
        )
        # endregion

        # Create config files
        for fname, txt in zip(['kilosort2_master.m', 'kilosort2_config.m', 'kilosort2_channelmap.m'], [kilosort2_master_txt, kilosort2_config_txt, kilosort2_channelmap_txt]):
            with (output_folder / fname).open('w') as f:
                f.write(txt)

        # Matlab (for reading and writing numpy) scripts texts
        writeNPY_text = '''% NPY-MATLAB writeNPY function. Copied from https://github.com/kwikteam/npy-matlab

function writeNPY(var, filename)
% function writeNPY(var, filename)
%
% Only writes little endian, fortran (column-major) ordering; only writes
% with NPY version number 1.0.
%
% Always outputs a shape according to matlab's convention, e.g. (10, 1)
% rather than (10,).

shape = size(var);
dataType = class(var);

header = constructNPYheader(dataType, shape);

fid = fopen(filename, 'w');
fwrite(fid, header, 'uint8');
fwrite(fid, var, dataType);
fclose(fid);


end'''
        constructNPYheader_text = '''% NPY-MATLAB constructNPYheader function. Copied from https://github.com/kwikteam/npy-matlab


function header = constructNPYheader(dataType, shape, varargin)

	if ~isempty(varargin)
		fortranOrder = varargin{1}; % must be true/false
		littleEndian = varargin{2}; % must be true/false
	else
		fortranOrder = true;
		littleEndian = true;
	end

    dtypesMatlab = {'uint8','uint16','uint32','uint64','int8','int16','int32','int64','single','double', 'logical'};
    dtypesNPY = {'u1', 'u2', 'u4', 'u8', 'i1', 'i2', 'i4', 'i8', 'f4', 'f8', 'b1'};

    magicString = uint8([147 78 85 77 80 89]); %x93NUMPY

    majorVersion = uint8(1);
    minorVersion = uint8(0);

    % build the dict specifying data type, array order, endianness, and
    % shape
    dictString = '{''descr'': ''\';

    if littleEndian
        dictString = [dictString '<'];
    else
        dictString = [dictString '>'];
    end

    dictString = [dictString dtypesNPY{strcmp(dtypesMatlab,dataType)} \''', '];

    dictString = [dictString \'''fortran_order'': '];

    if fortranOrder
        dictString = [dictString 'True, '];
    else
        dictString = [dictString 'False, '];
    end

    dictString = [dictString \'''shape'': ('];

%     if length(shape)==1 && shape==1
%
%     else
%         for s = 1:length(shape)
%             if s==length(shape) && shape(s)==1
%
%             else
%                 dictString = [dictString num2str(shape(s))];
%                 if length(shape)>1 && s+1==length(shape) && shape(s+1)==1
%                     dictString = [dictString ','];
%                 elseif length(shape)>1 && s<length(shape)
%                     dictString = [dictString ', '];
%                 end
%             end
%         end
%         if length(shape)==1
%             dictString = [dictString ','];
%         end
%     end

    for s = 1:length(shape)
        dictString = [dictString num2str(shape(s))];
        if s<length(shape)
            dictString = [dictString ', '];
        end
    end

    dictString = [dictString '), '];

    dictString = [dictString '}'];

    totalHeaderLength = length(dictString)+10; % 10 is length of magicString, version, and headerLength

    headerLengthPadded = ceil(double(totalHeaderLength+1)/16)*16; % the whole thing should be a multiple of 16
                                                                  % I add 1 to the length in order to allow for the newline character

	% format specification is that headerlen is little endian. I believe it comes out so using this command...
    headerLength = typecast(int16(headerLengthPadded-10), 'uint8');

    zeroPad = zeros(1,headerLengthPadded-totalHeaderLength, 'uint8')+uint8(32); % +32 so they are spaces
    zeroPad(end) = uint8(10); % newline character

    header = uint8([magicString majorVersion minorVersion headerLength dictString zeroPad]);

end'''

        # Create matlab scripts
        for fname, txt in zip(['writeNPY.m', 'constructNPYheader.m'], [writeNPY_text, constructNPYheader_text]):
            with (output_folder / fname).open('w') as f:
                f.write(txt)

    def start_sorting(self, output_folder, raise_error, verbose):
        output_folder = Path(output_folder)

        t0 = time.perf_counter()

        try:
            self.execute_kilosort_file(output_folder, verbose)
            t1 = time.perf_counter()
            run_time = float(t1 - t0)
            has_error = False
        except Exception as err:
            has_error = True
            run_time = None

        # Kilosort has a log file dir to shellscript launcher
        runtime_trace_path = output_folder / 'kilosort2.log'
        runtime_trace = []
        if runtime_trace_path.is_file():
            with open(runtime_trace_path, 'r') as fp:
                line = fp.readline()
                while line:
                    runtime_trace.append(line.strip())
                    line = fp.readline()

        if verbose:
            if has_error:
                print('Error running kilosort2')
            else:
                print(f'kilosort2 run time: {run_time:0.2f}s')

        if has_error and raise_error:
            # print(runtime_trace)
            raise Exception(
                f"Spike sorting failed. You can inspect the runtime trace in {output_folder}/kilosort2.log")

        return run_time

    @staticmethod
    def execute_kilosort_file(output_folder, verbose):
        print('Running kilosort file')

        os.environ['MW_NVCC_PATH'] = "/usr/local/cuda-10.1/bin"
        shell_cmd = f'''
                    #!/bin/bash
                    cd "{output_folder}"
                    module load MatLab/R2021b
                    module load cuda/10.1
                    matlab -nosplash -nodisplay -log -r kilosort2_master
                '''
        shell_script = ShellScript(shell_cmd, script_path=output_folder / 'run_kilosort2',
                                   log_path=output_folder / 'kilosort2.log', verbose=verbose)
        shell_script.start()
        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception('kilosort2 returned a non-zero exit code')

    def check_if_installed(self):
        if (Path(self.path) / 'master_kilosort.m').is_file() or (Path(self.path) / 'main_kilosort.m').is_file():
            return True
        else:
            return False

    @staticmethod
    def set_kilosort_path(kilosort_path):
        path = str(Path(kilosort_path).absolute())

        try:
            print("Setting KILOSORT2_PATH environment variable for subprocess calls to:", path)
            os.environ["KILOSORT2_PATH"] = path
        except Exception as e:
            print("Could not set KILOSORT2_PATH environment variable:", e)

        return path

    @staticmethod
    def format_params():
        if kilosort_params['NT'] is None:
            kilosort_params['NT'] = 64 * 1024 + kilosort_params['ntbuff']
        else:
            kilosort_params['NT'] = kilosort_params['NT'] // 32 * 32  # make sure is multiple of 32

        if kilosort_params['car']:
            kilosort_params['car'] = 1
        else:
            kilosort_params['car'] = 0

    @classmethod
    def get_result_from_folder(cls, output_folder):
        return KilosortSortingExtractor(folder_path=output_folder, keep_good_only=kilosort_params["keep_good_only"])


class KilosortSortingExtractor:
    """
    Represents data from Phy and Kilosort output folder as Python object

    Parameters
    ----------
    folder_path: str or Path
        Path to the output Phy folder (containing the params.py which stores data about the raw recording)
    exclude_cluster_groups: list or str (optional)
        Cluster groups to exclude (e.g. "noise" or ["noise", "mua"])
    """

    extractor_name = 'BasePhyKilosortSorting'
    installed = False  # check at class level if installed or not
    is_writable = False
    is_dumpable = True
    mode = 'folder'

    def __init__(self, folder_path, exclude_cluster_groups=None, keep_good_only=False):
        # Folder containing the numpy results of Kilosort
        phy_folder = Path(folder_path)
        self.folder = phy_folder.absolute()

        spike_times = np.load(str(phy_folder / 'spike_times.npy'))
        spike_clusters = np.load(str(phy_folder / 'spike_clusters.npy'))

        # The unit_ids with at least 1 spike
        unit_ids_with_spike = set(spike_clusters.flatten())

        spike_times.astype(int)
        params = Utils.read_python(str(phy_folder / 'params.py'))
        self.sampling_frequency = params['sample_rate']

        # Load properties from tsv/csv files
        all_property_files = [p for p in phy_folder.iterdir() if p.suffix in ['.csv', '.tsv']]

        cluster_info = None
        for file in all_property_files:
            if file.suffix == ".tsv":
                delimeter = "\t"
            else:
                delimeter = ","
            new_property = pd.read_csv(file, delimiter=delimeter)
            if cluster_info is None:
                cluster_info = new_property
            else:
                if new_property.columns[-1] not in cluster_info.columns:
                    # cluster_KSLabel.tsv and cluster_group.tsv are identical and have the same columns
                    # This prevents the same column data being added twice
                    cluster_info = pd.merge(cluster_info, new_property, on='cluster_id')

        # In case no tsv/csv files are found populate cluster info with minimal info
        if cluster_info is None:
            unit_ids_with_spike_list = list(unit_ids_with_spike)
            cluster_info = pd.DataFrame({'cluster_id': unit_ids_with_spike_list})
            cluster_info['group'] = ['unsorted'] * len(unit_ids_with_spike_list)

        # If pandas column for the unit_ids uses different name
        if "cluster_id" not in cluster_info.columns:
            assert "id" in cluster_info.columns, "Couldn't find cluster ids in the tsv files!"
            cluster_info["cluster_id"] = cluster_info["id"]
            del cluster_info["id"]

        if exclude_cluster_groups is not None:
            if isinstance(exclude_cluster_groups, str):
                cluster_info = cluster_info.query(f"group != '{exclude_cluster_groups}'")
            elif isinstance(exclude_cluster_groups, list):
                if len(exclude_cluster_groups) > 0:
                    for exclude_group in exclude_cluster_groups:
                        cluster_info = cluster_info.query(f"group != '{exclude_group}'")

        if keep_good_only and "KSLabel" in cluster_info.columns:
            cluster_info = cluster_info.query("KSLabel == 'good'")

        self._sorting_segments = []
        self.add_sorting_segment(KilosortSortingSegment(spike_times, spike_clusters))
        self._all_spikes = spike_times
        self._all_clusters = spike_clusters

        self._kwargs = {'folder_path': str(Path(folder_path).absolute()),
                        'keep_good_only': keep_good_only}

        all_unit_ids = cluster_info["cluster_id"].values
        self.unit_ids = []
        # Exclude units with 0 spikes
        for unit_id in all_unit_ids:
            if unit_id in unit_ids_with_spike:
                self.unit_ids.append(unit_id)

    def id_to_index(self, id):
        # Returns the index of the waveform unit id in the stored list
        return list(self.unit_ids).index(id)

    def get_num_segments(self):
        return len(self._sorting_segments)

    def add_sorting_segment(self, sorting_segment):
        self._sorting_segments.append(sorting_segment)
        sorting_segment.set_parent_extractor(self)

    def get_unit_spike_train(self,
                             unit_id,
                             segment_index: Union[int, None] = None,
                             start_frame: Union[int, None] = None,
                             end_frame: Union[int, None] = None,
                             ):
        if segment_index is None:
            if self.get_num_segments() == 1:
                segment_index = 0
            else:
                raise ValueError()

        segment = self._sorting_segments[segment_index]

        spike_train = segment.get_unit_spike_train(
            unit_id=unit_id, start_frame=start_frame, end_frame=end_frame).astype("int64")

        # np.atleast_1d is because caller expects iterable array
        return np.atleast_1d(spike_train)


class ShellScript:
    """
    Code for running shell scripts
    (used to run Kilosort2's MATLAB files from shell)
    """
    PathType = Union[str, Path]

    def __init__(self, script: str, script_path: Optional[PathType] = None, log_path: Optional[PathType] = None,
                 keep_temp_files: bool = False, verbose: bool = False):
        lines = script.splitlines()
        lines = self._remove_initial_blank_lines(lines)
        if len(lines) > 0:
            num_initial_spaces = self._get_num_initial_spaces(lines[0])
            for ii, line in enumerate(lines):
                if len(line.strip()) > 0:
                    n = self._get_num_initial_spaces(line)
                    if n < num_initial_spaces:
                        print(script)
                        raise Exception('Problem in script. First line must not be indented relative to others')
                    lines[ii] = lines[ii][num_initial_spaces:]
        self._script = '\n'.join(lines)
        self._script_path = script_path
        self._log_path = log_path
        self._keep_temp_files = keep_temp_files
        self._process: Optional[subprocess.Popen] = None
        self._files_to_remove: List[str] = []
        self._dirs_to_remove: List[str] = []
        self._start_time: Optional[float] = None
        self._verbose = verbose

    def __del__(self):
        self.cleanup()

    def substitute(self, old: str, new: Any) -> None:
        self._script = self._script.replace(old, '{}'.format(new))

    def write(self, script_path: Optional[str] = None) -> None:
        if script_path is None:
            script_path = self._script_path
        if script_path is None:
            raise Exception('Cannot write script. No path specified')
        with open(script_path, 'w') as f:
            f.write(self._script)
        os.chmod(script_path, 0o744)

    def start(self) -> None:
        if self._script_path is not None:
            script_path = Path(self._script_path)
            if script_path.suffix == '':
                if 'win' in sys.platform and sys.platform != 'darwin':
                    script_path = script_path.parent / (script_path.name + '.bat')
                else:
                    script_path = script_path.parent / (script_path.name + '.sh')
        else:
            tempdir = Path(tempfile.mkdtemp(prefix='tmp_shellscript'))
            if 'win' in sys.platform and sys.platform != 'darwin':
                script_path = tempdir / 'script.bat'
            else:
                script_path = tempdir / 'script.sh'
            self._dirs_to_remove.append(tempdir)

        if self._log_path is None:
            script_log_path = script_path.parent / 'spike_sorters_log.txt'
        else:
            script_log_path = Path(self._log_path)
            if script_path.suffix == '':
                script_log_path = script_log_path.parent / (script_log_path.name + '.txt')

        self.write(script_path)
        cmd = str(script_path)
        print('RUNNING SHELL SCRIPT: ' + cmd)
        self._start_time = time.time()
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                                         universal_newlines=True)
        with open(script_log_path, 'w+') as script_log_file:
            for line in self._process.stdout:
                script_log_file.write(line)
                if self._verbose:  # Print onto console depending on the verbose property passed on from the sorter class
                    print(line)

    def wait(self, timeout=None) -> Optional[int]:
        if not self.isRunning():
            return self.returnCode()
        assert self._process is not None, "Unexpected self._process is None even though it is running."
        try:
            retcode = self._process.wait(timeout=timeout)
            return retcode
        except:
            return None

    def cleanup(self) -> None:
        if self._keep_temp_files:
            return
        for dirpath in self._dirs_to_remove:
            ShellScript._rmdir_with_retries(str(dirpath), num_retries=5)

    def stop(self) -> None:
        if not self.isRunning():
            return
        assert self._process is not None, "Unexpected self._process is None even though it is running."

        signals = [signal.SIGINT] * 10 + [signal.SIGTERM] * 10 + [signal.SIGKILL] * 10

        for signal0 in signals:
            self._process.send_signal(signal0)
            try:
                self._process.wait(timeout=0.02)
                return
            except:
                pass

    def kill(self) -> None:
        if not self.isRunning():
            return

        assert self._process is not None, "Unexpected self._process is None even though it is running."
        self._process.send_signal(signal.SIGKILL)
        try:
            self._process.wait(timeout=1)
        except:
            print('WARNING: unable to kill shell script.')
            pass

    def stopWithSignal(self, sig, timeout) -> bool:
        if not self.isRunning():
            return True

        assert self._process is not None, "Unexpected self._process is None even though it is running."
        self._process.send_signal(sig)
        try:
            self._process.wait(timeout=timeout)
            return True
        except:
            return False

    def elapsedTimeSinceStart(self) -> Optional[float]:
        if self._start_time is None:
            return None

        return time.time() - self._start_time

    def isRunning(self) -> bool:
        if not self._process:
            return False
        retcode = self._process.poll()
        if retcode is None:
            return True
        return False

    def isFinished(self) -> bool:
        if not self._process:
            return False
        return not self.isRunning()

    def returnCode(self) -> Optional[int]:
        if not self.isFinished():
            raise Exception('Cannot get return code before process is finished.')
        assert self._process is not None, "Unexpected self._process is None even though it is finished."
        return self._process.returncode

    def scriptPath(self) -> Optional[str]:
        return self._script_path

    def _remove_initial_blank_lines(self, lines: List[str]) -> List[str]:
        ii = 0
        while ii < len(lines) and len(lines[ii].strip()) == 0:
            ii = ii + 1
        return lines[ii:]

    def _get_num_initial_spaces(self, line: str) -> int:
        ii = 0
        while ii < len(line) and line[ii] == ' ':
            ii = ii + 1
        return ii

    @staticmethod
    def _rmdir_with_retries(dirname, num_retries, delay_between_tries=1):
        for retry_num in range(1, num_retries + 1):
            if not os.path.exists(dirname):
                return
            try:
                shutil.rmtree(dirname)
                break
            except:
                if retry_num < num_retries:
                    print('Retrying to remove directory: {}'.format(dirname))
                    time.sleep(delay_between_tries)
                else:
                    raise Exception('Unable to remove directory after {} tries: {}'.format(num_retries, dirname))
# region Sorting Segments
class BaseSegment:
    """
    A segment of a recording or sorting output
    Needed to create weak references for garbage collection
    """

    def __init__(self):
        self._parent_extractor = None

    @property
    def parent_extractor(self):
        return self._parent_extractor()

    def set_parent_extractor(self, parent_extractor):
        self._parent_extractor = weakref.ref(parent_extractor)


class KilosortSortingSegment(BaseSegment):
    """
    A segment of the kilosort output
    """
    def __init__(self, all_spikes, all_clusters):
        BaseSegment.__init__(self)
        self._all_spikes = all_spikes
        self._all_clusters = all_clusters

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        spike_times = self._all_spikes[self._all_clusters == unit_id]
        if start_frame is not None:
            spike_times = spike_times[spike_times >= start_frame]
        if end_frame is not None:
            spike_times = spike_times[spike_times < end_frame]
        return np.atleast_1d(spike_times.copy().squeeze())
# endregion
# endregion


# region Extract Waveforms
class WaveformExtractor:
    def __init__(self, recording, sorting, folder):
        # region Sanity checks to make sure recording and sorting match
        assert recording.get_num_segments() == sorting.get_num_segments(), \
            "The recording and sorting objects must have the same number of segments!"

        sampling_frequency = recording.get_sampling_frequency()
        np.testing.assert_almost_equal(sampling_frequency, sorting.sampling_frequency, decimal=2)
        # endregion

        self.recording = recording
        self.sampling_frequency = sampling_frequency

        self.sorting = sorting
        self.folder = Path(folder)

        # Cache in memory
        self._waveforms = {}
        self.template_cache = {}
        self._extremum_amplitudes = None
        self._extremum_channels_indices = None

        # Set Parameters
        self.nbefore = self.ms_to_samples(ms_before)  # Number of samples before waveform peak to include
        self.nafter = self.ms_to_samples(ms_after)    # Number of samplesa after waveform peak to include
        self.nsamples = self.nbefore + self.nafter  # Total number of samples in waveform

        return_scaled_corrected = return_scaled
        if return_scaled_corrected:
            # Check if recording has scaled values:
            if not self.recording.has_scaled_traces():
                print("Setting 'return_scaled' to False")
                return_scaled_corrected = False
        self.return_scaled = return_scaled_corrected

        dtype = waveform_dtype
        if dtype is None:
            dtype = self.recording.get_dtype()
        if np.issubdtype(dtype, np.integer) and return_scaled_corrected:
            # If waveform will be scaled, dtype cannot be an integer (that would cause rounding)
            print("Setting 'dtype' to float32 because 'return_scaled' is True")
            dtype = "float32"
        dtype = np.dtype(dtype)
        self.dtype = dtype.str

    def ms_to_samples(self, ms):
        return int(ms * self.sampling_frequency / 1000.)

    # region Loading saved waveform extractor
    @classmethod
    def load_from_folder(cls, recording, sorting, folder):
        we = cls(recording, sorting, folder)

        _possible_template_modes = ('average', 'std', 'median')
        for mode in _possible_template_modes:
            # Load cached templates
            template_file = folder / f'templates_{mode}.npy'
            if template_file.is_file():
                we.template_cache[mode] = np.load(template_file)

        return we
    # endregion

    # region Creating new waveform extractor
    @classmethod
    def create(cls, recording, sorting, folder):
        folder = Path(folder)
        create_folder(folder / 'waveforms')
        return cls(recording, sorting, folder)
    # endregion

    # region Extract waveforms
    def run_extract_waveforms(self, **job_kwargs):
        num_chans = self.recording.get_num_channels()
        job_kwargs["n_jobs"] = Utils.ensure_n_jobs(self.recording, job_kwargs.get('n_jobs', None))

        selected_spikes = self.sample_spikes()

        # Get spike times
        selected_spike_times = {}
        for unit_id in self.sorting.unit_ids:
            selected_spike_times[unit_id] = []
            for segment_index in range(self.sorting.get_num_segments()):
                spike_times = self.sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                sel = selected_spikes[unit_id][segment_index]
                selected_spike_times[unit_id].append(spike_times[sel])

        # prepare memmap
        print("Preparing memory maps for waveforms")
        wfs_memmap = {}
        for unit_id in self.sorting.unit_ids:
            file_path = self.folder / 'waveforms' / f'waveforms_{unit_id}.npy'
            n_spikes = np.sum([e.size for e in selected_spike_times[unit_id]])
            shape = (n_spikes, self.nsamples, num_chans)
            wfs = np.zeros(shape, self.dtype)
            np.save(str(file_path), wfs)
            # wfs = np.load(file_path, mmap_mode='r+')
            wfs_memmap[unit_id] = file_path

        # and run
        func = WaveformExtractor._waveform_extractor_chunk
        init_func = WaveformExtractor._init_worker_waveform_extractor

        init_args = (self.recording, self.sorting,)
        # endregion

        init_args = init_args + (wfs_memmap, selected_spikes, selected_spike_times, self.nbefore, self.nafter, self.return_scaled)
        processor = ChunkRecordingExecutor(self.recording, func, init_func, init_args, job_name='extract waveforms',
                                           **job_kwargs)
        processor.run()

    def sample_spikes(self):
        """
        Uniform random selection of spikes per unit and save to .npy

        self.samples_spikes just calls self.random_spikes_uniformly and saves data to .npy files

        :return:
        Dictionary of {unit_id, [selected_spike_times]}
        """

        print("Sampling spikes for each unit")
        selected_spikes = self.select_random_spikes_uniformly()

        # Store in 2 columns (spike_index, segment_index) in a .npy file
        # NOT NECESSARY BUT COULD BE USEFUL FOR DEBUGGING
        print("Saving sampled spikes in .npy format")
        for unit_id in self.sorting.unit_ids:
            n = np.sum([e.size for e in selected_spikes[unit_id]])
            sampled_index = np.zeros(n, dtype=[('spike_index', 'int64'), ('segment_index', 'int64')])
            pos = 0
            for segment_index in range(self.sorting.get_num_segments()):
                inds = selected_spikes[unit_id][segment_index]
                sampled_index[pos:pos + inds.size]['spike_index'] = inds
                sampled_index[pos:pos + inds.size]['segment_index'] = segment_index
                pos += inds.size

            sampled_index_file = self.folder / 'waveforms' / f'sampled_index_{unit_id}.npy'
            np.save(str(sampled_index_file), sampled_index)

        return selected_spikes

    def select_random_spikes_uniformly(self):
        """
        Uniform random selection of spikes per unit.

        More complicated than necessary because it is designed to handle multi-segment data
        Must keep complications since ChunkRecordingExecutor expects multi-segment data

        :return:
        Dictionary of {unit_id, [selected_spike_times]}
        """
        sorting = self.sorting
        unit_ids = sorting.unit_ids
        num_seg = sorting.get_num_segments()

        selected_spikes = {}
        for unit_id in unit_ids:
            # spike per segment
            n_per_segment = [sorting.get_unit_spike_train(unit_id, segment_index=i).size for i in range(num_seg)]
            cum_sum = [0] + np.cumsum(n_per_segment).tolist()
            total = np.sum(n_per_segment)
            if max_waveforms_per_unit is not None:
                if total > max_waveforms_per_unit:
                    global_inds = np.random.choice(total, size=max_waveforms_per_unit, replace=False)
                    global_inds = np.sort(global_inds)
                else:
                    global_inds = np.arange(total)
            else:
                global_inds = np.arange(total)
            sel_spikes = []
            for segment_index in range(num_seg):
                in_segment = (global_inds >= cum_sum[segment_index]) & (global_inds < cum_sum[segment_index + 1])
                inds = global_inds[in_segment] - cum_sum[segment_index]

                # if max_waveforms_per_unit is not None:
                #     # clean border when sub selection
                #     assert nafter is not None
                #     spike_times = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                #     sampled_spike_times = spike_times[inds]
                #     num_samples = recording.get_num_samples(segment_index=segment_index)
                #     mask = (sampled_spike_times >= nbefore) & (sampled_spike_times < (num_samples - nafter))
                #     inds = inds[mask]

                sel_spikes.append(inds)
            selected_spikes[unit_id] = sel_spikes
        return selected_spikes

    @staticmethod
    def _waveform_extractor_chunk(segment_index, start_frame, end_frame, worker_ctx):
        # recover variables of the worker
        recording = worker_ctx['recording']
        sorting = worker_ctx['sorting']
        wfs_memmap_files = worker_ctx['wfs_memmap_files']
        selected_spikes = worker_ctx['selected_spikes']
        selected_spike_times = worker_ctx['selected_spike_times']
        nbefore = worker_ctx['nbefore']
        nafter = worker_ctx['nafter']
        return_scaled = worker_ctx['return_scaled']
        unit_cum_sum = worker_ctx['unit_cum_sum']

        seg_size = recording.get_num_samples(segment_index=segment_index)

        to_extract = {}
        for unit_id in sorting.unit_ids:
            spike_times = selected_spike_times[unit_id][segment_index]
            i0 = np.searchsorted(spike_times, start_frame)
            i1 = np.searchsorted(spike_times, end_frame)
            if i0 != i1:
                # protect from spikes on border :  spike_time<0 or spike_time>seg_size
                # useful only when max_spikes_per_unit is not None
                # waveform will not be extracted and a zeros will be left in the memmap file
                while (spike_times[i0] - nbefore) < 0 and (i0 != i1):
                    i0 = i0 + 1
                while (spike_times[i1 - 1] + nafter) > seg_size and (i0 != i1):
                    i1 = i1 - 1

            if i0 != i1:
                to_extract[unit_id] = i0, i1, spike_times[i0:i1]

        if len(to_extract) > 0:
            start = min(st[0] for _, _, st in to_extract.values()) - nbefore   # Get the minimum time frame from recording needed for extracting waveform from the minimum spike time - nbefore
            end = max(st[-1] for _, _, st in to_extract.values()) + nafter
            start = int(start)
            end = int(end)

            # load trace in memory
            traces = recording.get_traces(start_frame=start, end_frame=end, segment_index=segment_index,
                                          return_scaled=return_scaled)

            for unit_id, (i0, i1, local_spike_times) in to_extract.items():
                wfs = np.load(wfs_memmap_files[unit_id], mmap_mode="r+")
                for i in range(local_spike_times.size):
                    st = local_spike_times[i]
                    st = int(st)
                    pos = unit_cum_sum[unit_id][segment_index] + i0 + i
                    # st = spike time
                    # st - start = converts st from index position when whole recording is used to index position when only a chunk (trace) is used
                    wfs[pos, :, :] = traces[st - start - nbefore:st - start + nafter, :]  # Python slices with [start, end), so waveform is in format (nbefore + spike_location + nafter-1, n_channels)

    @staticmethod
    def _init_worker_waveform_extractor(recording, sorting, wfs_memmap,
                                        selected_spikes, selected_spike_times, nbefore, nafter, return_scaled):
        # create a local dict per worker
        worker_ctx = {}
        worker_ctx['recording'] = recording

        worker_ctx['sorting'] = sorting

        worker_ctx['wfs_memmap_files'] = wfs_memmap
        worker_ctx['selected_spikes'] = selected_spikes
        worker_ctx['selected_spike_times'] = selected_spike_times
        worker_ctx['nbefore'] = nbefore
        worker_ctx['nafter'] = nafter
        worker_ctx['return_scaled'] = return_scaled

        num_seg = sorting.get_num_segments()
        unit_cum_sum = {}
        for unit_id in sorting.unit_ids:
            # spike per segment
            n_per_segment = [selected_spikes[unit_id][i].size for i in range(num_seg)]
            cum_sum = [0] + np.cumsum(n_per_segment).tolist()
            unit_cum_sum[unit_id] = cum_sum
        worker_ctx['unit_cum_sum'] = unit_cum_sum

        return worker_ctx
    # endregion

    # region Get waveforms and templates
    def get_waveforms(self, unit_id, with_index=False, cache=True, memmap=True):
        """
        Return waveforms for the specified unit id.

        Parameters
        ----------
        unit_id: int or str
            Unit id to retrieve waveforms for
        with_index: bool
            If True, spike indices of extracted waveforms are returned (default False)
        cache: bool
            If True, waveforms are cached to the self.waveforms dictionary (default False)
        memmap: bool
            If True, waveforms are loaded as memmap objects.
            If False, waveforms are loaded as np.array objects (default True)

        Returns
        -------
        wfs: np.array
            The returned waveform (num_spikes, num_samples, num_channels)
            num_samples = nbefore + 1 (for value at peak) + nafter-1
        indices: np.array
            If 'with_index' is True, the spike indices corresponding to the waveforms extracted
        """
        assert unit_id in self.sorting.unit_ids, "'unit_id' is invalid"
        wfs = self._waveforms.get(unit_id, None)
        if wfs is None:
            waveform_file = self.folder / 'waveforms' / f'waveforms_{unit_id}.npy'
            if not waveform_file.is_file():
                raise Exception('Waveforms not extracted yet: '
                                'please do WaveformExtractor.run_extract_waveforms() first')
            if memmap:
                wfs = np.load(waveform_file, mmap_mode="r")
            else:
                wfs = np.load(waveform_file)
            if cache:
                self._waveforms[unit_id] = wfs

        if with_index:
            sampled_index = self.get_sampled_indices(unit_id)
            return wfs, sampled_index
        else:
            return wfs

    def get_sampled_indices(self, unit_id):
        """
        Return sampled spike indices of extracted waveforms
        (which waveforms correspond to which spikes if "max_spikes_per_unit" is not None)

        Parameters
        ----------
        unit_id: int
            Unit id to retrieve indices for

        Returns
        -------
        sampled_indices: np.array
            The sampled indices with shape (n_waveforms,)
        """

        sampled_index_file = self.folder / 'waveforms' / f"sampled_index_{unit_id}.npy"
        sampled_index = np.load(str(sampled_index_file))

        # When this function was written, the sampled_index .npy files also included segment index of spikes
        # This disregards segment index since there should only be 1 segment
        sampled_index_without_segment_index = []
        for index in sampled_index:
            sampled_index_without_segment_index.append(index[0])
        return sampled_index_without_segment_index

    def get_template(self, unit_id, mode='average'):
        """
        Return template (average waveform).

        Parameters
        ----------
        unit_id: int
            Unit id to retrieve waveforms for
        mode: str
            'average' (default), 'median' , 'std'(standard deviation)
        sparsity: dict or None
            If given, dictionary with unit ids as keys and channel sparsity by index as values.
            The sparsity can be computed with the toolkit.get_template_channel_sparsity() function
            (make sure to use the default output='id' when computing the sparsity)

        Returns
        -------
        template: np.array
            The returned template (num_samples, num_channels)
        """

        _possible_template_modes = ('average', 'std', 'median')
        assert mode in _possible_template_modes
        assert unit_id in self.sorting.unit_ids

        if mode in self.template_cache:
            # already in the global cache
            unit_ind = self.sorting.id_to_index(unit_id)
            template = self.template_cache[mode][unit_ind, :, :]
            return template

        # compute from waveforms
        wfs = self.get_waveforms(unit_id)
        if mode == 'median':
            template = np.median(wfs, axis=0)
        elif mode == 'average':
            template = np.average(wfs, axis=0)
        elif mode == 'std':
            template = np.std(wfs, axis=0)
        return template

    def compute_templates(self, modes=('average', 'std')):
        """
        Compute all template for different "modes":
          * average
          * std
          * median

        The results are cached in memory as 3d ndarray (nunits, nsamples, nchans)
        and also saved as npy file in the folder to avoid recomputation each time.
        """
        # TODO : run this in parralel

        print_stage("COMPUTING TEMPLATES")
        print("Template modes: " + ", ".join(modes))
        stopwatch = Stopwatch()
        unit_ids = self.sorting.unit_ids
        num_chans = self.recording.get_num_channels()

        for mode in modes:
            templates = np.zeros((len(unit_ids), self.nsamples, num_chans), dtype=self.dtype)
            self.template_cache[mode] = templates

        for i, unit_id in enumerate(unit_ids):
            wfs = self.get_waveforms(unit_id, cache=False)
            for mode in modes:
                if mode == 'median':
                    arr = np.median(wfs, axis=0)
                elif mode == 'average':
                    arr = np.average(wfs, axis=0)
                elif mode == 'std':
                    arr = np.std(wfs, axis=0)
                else:
                    raise ValueError('mode must in median/average/std')

                self.template_cache[mode][i, :, :] = arr

        print("Saving templates to .npy")
        for mode in modes:
            templates = self.template_cache[mode]
            template_file = self.folder / f'templates_{mode}.npy'
            np.save(str(template_file), templates)
        stopwatch.log_time("Done computing and saving templates.")

    def get_template_amplitudes(self, peak_sign='neg'):
        """
        Get amplitude per channel for each unit.

        Parameters
        ----------
        peak_sign: str
            Sign of the template to compute amplitude ('neg', 'pos', 'both')

        Returns
        -------
        peak_values: dict
            Dictionary with unit ids as keys and template amplitudes as values
        """
        assert peak_sign in ('both', 'neg', 'pos')

        peak_values = {}
        for unit_id in self.sorting.unit_ids:
            template = self.get_template(unit_id, mode='average')

            if peak_sign == 'both':
                values = np.max(np.abs(template), axis=0)
            elif peak_sign == 'neg':
                values = -np.min(template, axis=0)
            elif peak_sign == 'pos':
                values = np.max(template, axis=0)

            peak_values[unit_id] = np.abs(values)
        return peak_values

    def get_template_extremums(self, cache=True, peak_sign='neg'):
        """
        Compute the amplitude and the channel with the extremum peak for each unit in self.sorting.unit_ids.

        Parameters
        ----------
        cache: bool
            If True, store results in memory
        peak_sign: str
            Sign of the template to compute amplitude and best channels ('neg', 'pos', 'both')

        Returns
        -------
        extremum_amplitudes: dict
            Dictionary with unit_ids as keys and amplitudes as values
        extremum_channels_indices: dict
            Dictionary with unit ids as keys and extremum channels (indices) as values
        """
        if self._extremum_amplitudes is not None and self._extremum_channels_indices is not None:
            return self._extremum_amplitudes, self._extremum_channels_indices

        amplitudes = self.get_template_amplitudes(peak_sign=peak_sign)
        extremum_amplitudes = {}
        extremum_channels_indices = {}
        for unit_id in self.sorting.unit_ids:
            unit_amplitudes = amplitudes[unit_id]
            extremum_amplitudes[unit_id] = np.max(unit_amplitudes)
            extremum_channels_indices[unit_id] = np.argmax(unit_amplitudes)

        if cache:
            self._extremum_amplitudes = extremum_amplitudes
            self._extremum_channels_indices = extremum_channels_indices

        return extremum_amplitudes, extremum_channels_indices

    @staticmethod
    def get_peak_ind(waveforms, peak_sign='neg'):
        """
        Get the indices of the peaks in the waveforms

        Parameters
        ----------
        waveforms: np.array
            With shape (n_waveforms, n_timepoints)
        peak_sign: str
            Sign of the template to compute amplitude and best channels ('neg', 'pos', 'both')

        Returns
        -------
        np.array with shape (n_waveforms,) giving the peak index of each waveform

        """
        if peak_sign == 'both':
            peak_values_pos = np.max(waveforms, axis=1)
            peak_values_neg = np.min(waveforms, axis=1)
            peak_values = np.where(peak_values_pos > np.abs(peak_values_neg), peak_values_pos, peak_values_neg)
        elif peak_sign == 'neg':
            peak_values = np.min(waveforms, axis=1)
        elif peak_sign == 'pos':
            peak_values = np.max(waveforms, axis=1)

        peak_mask = np.repeat(peak_values[:, np.newaxis], waveforms.shape[1], axis=1)
        waveforms_peaks = waveforms == peak_mask
        return np.asarray([(indices := np.flatnonzero(peaks))[indices.size//2] for peaks in waveforms_peaks])

    def save_curated_units(self, unit_ids, curated_folder):
        """
        Filters units by storing curated units in a new folder.

        Parameters
        ----------
        unit_ids : list or array
            The unit ids to keep in the new WaveformExtractor object
        curated_folder : Path
            The new folder where curated waveforms are copied

        Return
        ------
        we :  WaveformExtractor
            The newly create waveform extractor with the selected units
        """
        print_stage("SAVING CURATED UNITS")
        stopwatch = Stopwatch()

        print("Saving curated units to new folder")
        create_folder(curated_folder)

        # Save unit_ids
        np.save(str(curated_folder/"unit_ids.npy"), unit_ids)

        # create and populate waveforms folder
        curated_waveforms_folder = curated_folder / "waveforms"
        create_folder(curated_waveforms_folder)

        waveforms_files = [f for f in (self.folder / "waveforms").iterdir() if f.suffix == ".npy"]
        for unit in unit_ids:
            for wf_file in waveforms_files:
                if f"waveforms_{unit}.npy" in wf_file.name or f'sampled_index_{unit}.npy' in wf_file.name:
                    shutil.copyfile(wf_file, curated_waveforms_folder / wf_file.name)

        we = WaveformExtractor.load_from_folder(self.recording, self.sorting, curated_folder)
        stopwatch.log_time("Done saving curated units.")

        return we

    def load_unit_ids(self):
        unit_ids_folder = self.folder / "unit_ids.npy"

        assert unit_ids_folder.exists(), "Cannot load curated waveform unit_ids since they have not been computed and saved yet"

        return np.load(unit_ids_folder)
    # endregion


class ChunkRecordingExecutor:
    """
    Used to extract waveforms from recording

    Core class for parallel processing to run a "function" over chunks on a recording.

    It supports running a function:
        * in loop with chunk processing (low RAM usage)
        * at once if chunk_size is None (high RAM usage)
        * in parallel with ProcessPoolExecutor (higher speed)

    The initializer ('init_func') allows to set a global context to avoid heavy serialization
    (for examples, see implementation in `core.WaveformExtractor`).

    Parameters
    ----------
    recording: RecordingExtractor
        The recording to be processed
    func: function
        Function that runs on each chunk
    init_func: function
        Initializer function to set the global context (accessible by 'func')
    init_args: tuple
        Arguments for init_func
    verbose: bool
        If True, output is verbose
    progress_bar: bool
        If True, a progress bar is printed to monitor the progress of the process
    handle_returns: bool
        If True, the function can return values
    n_jobs: int
        Number of jobs to be used (default 1). Use -1 to use as many jobs as number of cores
    total_memory: str
        Total memory (RAM) to use (e.g. "1G", "500M")
    chunk_memory: str
        Memory per chunk (RAM) to use (e.g. "1G", "500M")
    chunk_size: int or None
        Size of each chunk in number of samples. If 'total_memory' or 'chunk_memory' are used, it is ignored.
    job_name: str
        Job name

    Returns
    -------
    res: list
        If 'handle_returns' is True, the results for each chunk process
    """

    def __init__(self, recording, func, init_func, init_args, verbose=True, progress_bar=False, handle_returns=False,
                 n_jobs=1, total_memory=None, chunk_size=None, chunk_memory=None,
                 job_name=''):

        self.recording = recording
        self.func = func
        self.init_func = init_func
        self.init_args = init_args

        self.verbose = verbose
        self.progress_bar = progress_bar

        self.handle_returns = handle_returns

        self.n_jobs = Utils.ensure_n_jobs(recording, n_jobs=n_jobs)
        self.chunk_size = Utils.ensure_chunk_size(recording,
                                            total_memory=total_memory, chunk_size=chunk_size,
                                            chunk_memory=chunk_memory, n_jobs=self.n_jobs)
        self.job_name = job_name

        if verbose:
            print(self.job_name, 'with', 'n_jobs', self.n_jobs, ' chunk_size', self.chunk_size)

    def run(self):
        """
        Runs the defined jobs.
        """
        all_chunks = ChunkRecordingExecutor.divide_recording_into_chunks(self.recording, self.chunk_size)

        if self.handle_returns:
            returns = []
        else:
            returns = None

        import sys
        if self.n_jobs != 1 and not (sys.version_info >= (3, 8)):
            self.n_jobs = 1

        if self.n_jobs == 1:
            if self.progress_bar:
                all_chunks = tqdm(all_chunks, ascii=True, desc=self.job_name)

            worker_ctx = self.init_func(*self.init_args)
            for segment_index, frame_start, frame_stop in all_chunks:
                res = self.func(segment_index, frame_start, frame_stop, worker_ctx)
                if self.handle_returns:
                    returns.append(res)
        else:
            n_jobs = min(self.n_jobs, len(all_chunks))

            ######## Do you want to limit the number of threads per process?
            ######## It has to be done to speed up numpy a lot if multicores
            ######## Otherwise, np.dot will be slow. How to do that, up to you
            ######## This is just a suggestion, but here it adds a dependency

            # parallel
            with ProcessPoolExecutor(max_workers=n_jobs,
                                     initializer=ChunkRecordingExecutor.worker_initializer,
                                     initargs=(self.func, self.init_func, self.init_args)) as executor:

                # Results not needed
                results = executor.map(ChunkRecordingExecutor.function_wrapper, all_chunks)

                if self.progress_bar:
                    results = tqdm(results, desc=self.job_name, total=len(all_chunks))

                if self.handle_returns:  # Should be false
                    for res in results:
                        returns.append(res)
                else:
                    for res in results:
                        pass

        return returns

    @staticmethod
    def function_wrapper(args):
        segment_index, start_frame, end_frame = args
        global _func
        global _worker_ctx
        return _func(segment_index, start_frame, end_frame, _worker_ctx)

    @staticmethod
    def divide_recording_into_chunks(recording, chunk_size):
        all_chunks = []
        for segment_index in range(recording.get_num_segments()):
            num_frames = recording.get_num_samples(segment_index)
            chunks = ChunkRecordingExecutor.divide_segment_into_chunks(num_frames, chunk_size)
            all_chunks.extend([(segment_index, frame_start, frame_stop) for frame_start, frame_stop in chunks])
        return all_chunks

    @staticmethod
    def divide_segment_into_chunks(num_frames, chunk_size):
        if chunk_size is None:
            chunks = [(0, num_frames)]
        else:
            n = num_frames // chunk_size

            frame_starts = np.arange(n) * chunk_size
            frame_stops = frame_starts + chunk_size

            frame_starts = frame_starts.tolist()
            frame_stops = frame_stops.tolist()

            if (num_frames % chunk_size) > 0:
                frame_starts.append(n * chunk_size)
                frame_stops.append(num_frames)

            chunks = list(zip(frame_starts, frame_stops))

        return chunks

    @staticmethod
    def worker_initializer(func, init_func, init_args):
        global _worker_ctx
        _worker_ctx = init_func(*init_args)
        global _func
        _func = func
global _worker_ctx
global _func
# region ProcessPoolExecutor
# Used for parallel processing in ChunkRecordingExecutor
# Copyright 2009 Brian Quinlan. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Implements ProcessPoolExecutor.

The following diagram and text describe the data-flow through the system:

|======================= In-process =====================|== Out-of-process ==|

+----------+     +----------+       +--------+     +-----------+    +---------+
|          |  => | Work Ids |       |        |     | Call Q    |    | Process |
|          |     +----------+       |        |     +-----------+    |  Pool   |
|          |     | ...      |       |        |     | ...       |    +---------+
|          |     | 6        |    => |        |  => | 5, call() | => |         |
|          |     | 7        |       |        |     | ...       |    |         |
| Process  |     | ...      |       | Local  |     +-----------+    | Process |
|  Pool    |     +----------+       | Worker |                      |  #1..n  |
| Executor |                        | Thread |                      |         |
|          |     +----------- +     |        |     +-----------+    |         |
|          | <=> | Work Items | <=> |        | <=  | Result Q  | <= |         |
|          |     +------------+     |        |     +-----------+    |         |
|          |     | 6: call()  |     |        |     | ...       |    |         |
|          |     |    future  |     |        |     | 4, result |    |         |
|          |     | ...        |     |        |     | 3, except |    |         |
+----------+     +------------+     +--------+     +-----------+    +---------+

Executor.submit() called:
- creates a uniquely numbered _WorkItem and adds it to the "Work Items" dict
- adds the id of the _WorkItem to the "Work Ids" queue

Local worker thread:
- reads work ids from the "Work Ids" queue and looks up the corresponding
  WorkItem from the "Work Items" dict: if the work item has been cancelled then
  it is simply removed from the dict, otherwise it is repackaged as a
  _CallItem and put in the "Call Q". New _CallItems are put in the "Call Q"
  until "Call Q" is full. NOTE: the size of the "Call Q" is kept small because
  calls placed in the "Call Q" can no longer be cancelled with Future.cancel().
- reads _ResultItems from "Result Q", updates the future stored in the
  "Work Items" dict and deletes the dict entry

Process #1..n:
- reads _CallItems from "Call Q", executes the calls, and puts the resulting
  _ResultItems in "Result Q"
"""

__author__ = 'Brian Quinlan (brian@sweetapp.com)'

import atexit
import os
from concurrent.futures import _base
import queue
from queue import Full
import multiprocessing as mp
import multiprocessing.connection
from multiprocessing.queues import Queue
import threading
import weakref
from functools import partial
import itertools
import sys
import traceback

# Workers are created as daemon threads and processes. This is done to allow the
# interpreter to exit when there are still idle processes in a
# ProcessPoolExecutor's process pool (i.e. shutdown() was not called). However,
# allowing workers to die with the interpreter has two undesirable properties:
#   - The workers would still be running during interpreter shutdown,
#     meaning that they would fail in unpredictable ways.
#   - The workers could be killed while evaluating a work item, which could
#     be bad if the callable being evaluated has external side-effects e.g.
#     writing to a file.
#
# To work around this problem, an exit handler is installed which tells the
# workers to exit when their work queues are empty and then waits until the
# threads/processes finish.

_threads_wakeups = weakref.WeakKeyDictionary()
_global_shutdown = False


class _ThreadWakeup:
    def __init__(self):
        self._reader, self._writer = mp.Pipe(duplex=False)

    def close(self):
        self._writer.close()
        self._reader.close()

    def wakeup(self):
        self._writer.send_bytes(b"")

    def clear(self):
        while self._reader.poll():
            self._reader.recv_bytes()


def _python_exit():
    global _global_shutdown
    _global_shutdown = True
    items = list(_threads_wakeups.items())
    for _, thread_wakeup in items:
        thread_wakeup.wakeup()
    for t, _ in items:
        t.join()

# Controls how many more calls than processes will be queued in the call queue.
# A smaller number will mean that processes spend more time idle waiting for
# work while a larger number will make Future.cancel() succeed less frequently
# (Futures in the call queue cannot be cancelled).
EXTRA_QUEUED_CALLS = 1


# On Windows, WaitForMultipleObjects is used to wait for processes to finish.
# It can wait on, at most, 63 objects. There is an overhead of two objects:
# - the result queue reader
# - the thread wakeup reader
_MAX_WINDOWS_WORKERS = 63 - 2

# Hack to embed stringification of remote traceback in local traceback

class _RemoteTraceback(Exception):
    def __init__(self, tb):
        self.tb = tb
    def __str__(self):
        return self.tb

class _ExceptionWithTraceback:
    def __init__(self, exc, tb):
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = ''.join(tb)
        self.exc = exc
        self.tb = '\n"""\n%s"""' % tb
    def __reduce__(self):
        return _rebuild_exc, (self.exc, self.tb)

def _rebuild_exc(exc, tb):
    exc.__cause__ = _RemoteTraceback(tb)
    return exc

class _WorkItem(object):
    def __init__(self, future, fn, args, kwargs):
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

class _ResultItem(object):
    def __init__(self, work_id, exception=None, result=None):
        self.work_id = work_id
        self.exception = exception
        self.result = result

class _CallItem(object):
    def __init__(self, work_id, fn, args, kwargs):
        self.work_id = work_id
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


class _SafeQueue(Queue):
    """Safe Queue set exception to the future object linked to a job"""
    def __init__(self, max_size=0, *, ctx, pending_work_items):
        self.pending_work_items = pending_work_items
        super().__init__(max_size, ctx=ctx)

    def _on_queue_feeder_error(self, e, obj):
        if isinstance(obj, _CallItem):
            tb = traceback.format_exception(type(e), e, e.__traceback__)
            e.__cause__ = _RemoteTraceback('\n"""\n{}"""'.format(''.join(tb)))
            work_item = self.pending_work_items.pop(obj.work_id, None)
            # work_item can be None if another process terminated. In this case,
            # the queue_manager_thread fails all work_items with BrokenProcessPool
            if work_item is not None:
                work_item.future.set_exception(e)
        else:
            super()._on_queue_feeder_error(e, obj)


def _get_chunks(*iterables, chunksize):
    """ Iterates over zip()ed iterables in chunks. """
    it = zip(*iterables)
    while True:
        chunk = tuple(itertools.islice(it, chunksize))
        if not chunk:
            return
        yield chunk

def _process_chunk(fn, chunk):
    """ Processes a chunk of an iterable passed to map.

    Runs the function passed to map() on a chunk of the
    iterable passed to map.

    This function is run in a separate process.

    """
    return [fn(*args) for args in chunk]


def _sendback_result(result_queue, work_id, result=None, exception=None):
    """Safely send back the given result or exception"""
    try:
        result_queue.put(_ResultItem(work_id, result=result,
                                     exception=exception))
    except BaseException as e:
        exc = _ExceptionWithTraceback(e, e.__traceback__)
        result_queue.put(_ResultItem(work_id, exception=exc))


def _process_worker(call_queue, result_queue, initializer, initargs):
    """Evaluates calls from call_queue and places the results in result_queue.

    This worker is run in a separate process.

    Args:
        call_queue: A ctx.Queue of _CallItems that will be read and
            evaluated by the worker.
        result_queue: A ctx.Queue of _ResultItems that will written
            to by the worker.
        initializer: A callable initializer, or None
        initargs: A tuple of args for the initializer
    """
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            _base.LOGGER.critical('Exception in initializer:', exc_info=True)
            # The parent will notice that the process stopped and
            # mark the pool broken
            return
    while True:
        call_item = call_queue.get(block=True)
        if call_item is None:
            # Wake up queue management thread
            result_queue.put(os.getpid())
            return
        try:
            r = call_item.fn(*call_item.args, **call_item.kwargs)
        except BaseException as e:
            exc = _ExceptionWithTraceback(e, e.__traceback__)
            _sendback_result(result_queue, call_item.work_id, exception=exc)
        else:
            _sendback_result(result_queue, call_item.work_id, result=r)
            del r

        # Liberate the resource as soon as possible, to avoid holding onto
        # open files or shared memory that is not needed anymore
        del call_item


def _add_call_item_to_queue(pending_work_items,
                            work_ids,
                            call_queue):
    """Fills call_queue with _WorkItems from pending_work_items.

    This function never blocks.

    Args:
        pending_work_items: A dict mapping work ids to _WorkItems e.g.
            {5: <_WorkItem...>, 6: <_WorkItem...>, ...}
        work_ids: A queue.Queue of work ids e.g. Queue([5, 6, ...]). Work ids
            are consumed and the corresponding _WorkItems from
            pending_work_items are transformed into _CallItems and put in
            call_queue.
        call_queue: A multiprocessing.Queue that will be filled with _CallItems
            derived from _WorkItems.
    """
    while True:
        if call_queue.full():
            return
        try:
            work_id = work_ids.get(block=False)
        except queue.Empty:
            return
        else:
            work_item = pending_work_items[work_id]

            if work_item.future.set_running_or_notify_cancel():
                call_queue.put(_CallItem(work_id,
                                         work_item.fn,
                                         work_item.args,
                                         work_item.kwargs),
                               block=True)
            else:
                del pending_work_items[work_id]
                continue


def _queue_management_worker(executor_reference,
                             processes,
                             pending_work_items,
                             work_ids_queue,
                             call_queue,
                             result_queue,
                             thread_wakeup):
    """Manages the communication between this process and the worker processes.

    This function is run in a local thread.

    Args:
        executor_reference: A weakref.ref to the ProcessPoolExecutor that owns
            this thread. Used to determine if the ProcessPoolExecutor has been
            garbage collected and that this function can exit.
        process: A list of the ctx.Process instances used as
            workers.
        pending_work_items: A dict mapping work ids to _WorkItems e.g.
            {5: <_WorkItem...>, 6: <_WorkItem...>, ...}
        work_ids_queue: A queue.Queue of work ids e.g. Queue([5, 6, ...]).
        call_queue: A ctx.Queue that will be filled with _CallItems
            derived from _WorkItems for processing by the process workers.
        result_queue: A ctx.SimpleQueue of _ResultItems generated by the
            process workers.
        thread_wakeup: A _ThreadWakeup to allow waking up the
            queue_manager_thread from the main Thread and avoid deadlocks
            caused by permanently locked queues.
    """
    executor = None

    def shutting_down():
        return (_global_shutdown or executor is None
                or executor._shutdown_thread)

    def shutdown_worker():
        # This is an upper bound on the number of children alive.
        n_children_alive = sum(p.is_alive() for p in processes.values())
        n_children_to_stop = n_children_alive
        n_sentinels_sent = 0
        # Send the right number of sentinels, to make sure all children are
        # properly terminated.
        while n_sentinels_sent < n_children_to_stop and n_children_alive > 0:
            for i in range(n_children_to_stop - n_sentinels_sent):
                try:
                    call_queue.put_nowait(None)
                    n_sentinels_sent += 1
                except Full:
                    break
            n_children_alive = sum(p.is_alive() for p in processes.values())

        # Release the queue's resources as soon as possible.
        call_queue.close()
        # If .join() is not called on the created processes then
        # some ctx.Queue methods may deadlock on Mac OS X.
        for p in processes.values():
            p.join()

    result_reader = result_queue._reader
    wakeup_reader = thread_wakeup._reader
    readers = [result_reader, wakeup_reader]

    while True:
        _add_call_item_to_queue(pending_work_items,
                                work_ids_queue,
                                call_queue)

        # Wait for a result to be ready in the result_queue while checking
        # that all worker processes are still running, or for a wake up
        # signal send. The wake up signals come either from new tasks being
        # submitted, from the executor being shutdown/gc-ed, or from the
        # shutdown of the python interpreter.
        worker_sentinels = [p.sentinel for p in processes.values()]
        ready = mp.connection.wait(readers + worker_sentinels)

        cause = None
        is_broken = True
        if result_reader in ready:
            try:
                result_item = result_reader.recv()
                is_broken = False
            except BaseException as e:
                cause = traceback.format_exception(type(e), e, e.__traceback__)

        elif wakeup_reader in ready:
            is_broken = False
            result_item = None
        thread_wakeup.clear()
        if is_broken:
            # Mark the process pool broken so that submits fail right now.
            executor = executor_reference()
            if executor is not None:
                executor._broken = ('A child process terminated '
                                    'abruptly, the process pool is not '
                                    'usable anymore')
                executor._shutdown_thread = True
                executor = None
            bpe = BrokenProcessPool("A process in the process pool was "
                                    "terminated abruptly while the future was "
                                    "running or pending.")
            if cause is not None:
                bpe.__cause__ = _RemoteTraceback(
                    f"\n'''\n{''.join(cause)}'''")
            # All futures in flight must be marked failed
            for work_id, work_item in pending_work_items.items():
                work_item.future.set_exception(bpe)
                # Delete references to object. See issue16284
                del work_item
            pending_work_items.clear()
            # Terminate remaining workers forcibly: the queues or their
            # locks may be in a dirty state and block forever.
            for p in processes.values():
                p.terminate()
            shutdown_worker()
            return
        if isinstance(result_item, int):
            # Clean shutdown of a worker using its PID
            # (avoids marking the executor broken)
            assert shutting_down()
            p = processes.pop(result_item)
            p.join()
            if not processes:
                shutdown_worker()
                return
        elif result_item is not None:
            work_item = pending_work_items.pop(result_item.work_id, None)
            # work_item can be None if another process terminated (see above)
            if work_item is not None:
                if result_item.exception:
                    work_item.future.set_exception(result_item.exception)
                else:
                    work_item.future.set_result(result_item.result)
                # Delete references to object. See issue16284
                del work_item
            # Delete reference to result_item
            del result_item

        # Check whether we should start shutting down.
        executor = executor_reference()
        # No more work items can be added if:
        #   - The interpreter is shutting down OR
        #   - The executor that owns this worker has been collected OR
        #   - The executor that owns this worker has been shutdown.
        if shutting_down():
            try:
                # Flag the executor as shutting down as early as possible if it
                # is not gc-ed yet.
                if executor is not None:
                    executor._shutdown_thread = True
                # Since no new work items can be added, it is safe to shutdown
                # this thread if there are no pending work items.
                if not pending_work_items:
                    shutdown_worker()
                    return
            except Full:
                # This is not a problem: we will eventually be woken up (in
                # result_queue.get()) and be able to send a sentinel again.
                pass
        executor = None


_system_limits_checked = False
_system_limited = None


def _check_system_limits():
    global _system_limits_checked, _system_limited
    if _system_limits_checked:
        if _system_limited:
            raise NotImplementedError(_system_limited)
    _system_limits_checked = True
    try:
        nsems_max = os.sysconf("SC_SEM_NSEMS_MAX")
    except (AttributeError, ValueError):
        # sysconf not available or setting not available
        return
    if nsems_max == -1:
        # indetermined limit, assume that limit is determined
        # by available memory only
        return
    if nsems_max >= 256:
        # minimum number of semaphores available
        # according to POSIX
        return
    _system_limited = ("system provides too few semaphores (%d"
                       " available, 256 necessary)" % nsems_max)
    raise NotImplementedError(_system_limited)


def _chain_from_iterable_of_lists(iterable):
    """
    Specialized implementation of itertools.chain.from_iterable.
    Each item in *iterable* should be a list.  This function is
    careful not to keep references to yielded objects.
    """
    for element in iterable:
        element.reverse()
        while element:
            yield element.pop()


class BrokenProcessPool(_base.BrokenExecutor):
    """
    Raised when a process in a ProcessPoolExecutor terminated abruptly
    while a future was in the running state.
    """


class ProcessPoolExecutor(_base.Executor):
    def __init__(self, max_workers=None, mp_context=None,
                 initializer=None, initargs=()):
        """Initializes a new ProcessPoolExecutor instance.

        Args:
            max_workers: The maximum number of processes that can be used to
                execute the given calls. If None or not given then as many
                worker processes will be created as the machine has processors.
            mp_context: A multiprocessing context to launch the workers. This
                object should provide SimpleQueue, Queue and Process.
            initializer: A callable used to initialize worker processes.
            initargs: A tuple of arguments to pass to the initializer.
        """
        _check_system_limits()

        if max_workers is None:
            self._max_workers = os.cpu_count() or 1
            if sys.platform == 'win32':
                self._max_workers = min(_MAX_WINDOWS_WORKERS,
                                        self._max_workers)
        else:
            if max_workers <= 0:
                raise ValueError("max_workers must be greater than 0")
            elif (sys.platform == 'win32' and
                max_workers > _MAX_WINDOWS_WORKERS):
                raise ValueError(
                    f"max_workers must be <= {_MAX_WINDOWS_WORKERS}")

            self._max_workers = max_workers

        if mp_context is None:
            mp_context = mp.get_context()
        self._mp_context = mp_context

        if initializer is not None and not callable(initializer):
            raise TypeError("initializer must be a callable")
        self._initializer = initializer
        self._initargs = initargs

        # Management thread
        self._queue_management_thread = None

        # Map of pids to processes
        self._processes = {}

        # Shutdown is a two-step process.
        self._shutdown_thread = False
        self._shutdown_lock = threading.Lock()
        self._broken = False
        self._queue_count = 0
        self._pending_work_items = {}

        # Create communication channels for the executor
        # Make the call queue slightly larger than the number of processes to
        # prevent the worker processes from idling. But don't make it too big
        # because futures in the call queue cannot be cancelled.
        queue_size = self._max_workers + EXTRA_QUEUED_CALLS
        self._call_queue = _SafeQueue(
            max_size=queue_size, ctx=self._mp_context,
            pending_work_items=self._pending_work_items)
        # Killed worker processes can produce spurious "broken pipe"
        # tracebacks in the queue's own worker thread. But we detect killed
        # processes anyway, so silence the tracebacks.
        self._call_queue._ignore_epipe = True
        self._result_queue = mp_context.SimpleQueue()
        self._work_ids = queue.Queue()

        # _ThreadWakeup is a communication channel used to interrupt the wait
        # of the main loop of queue_manager_thread from another thread (e.g.
        # when calling executor.submit or executor.shutdown). We do not use the
        # _result_queue to send the wakeup signal to the queue_manager_thread
        # as it could result in a deadlock if a worker process dies with the
        # _result_queue write lock still acquired.
        self._queue_management_thread_wakeup = _ThreadWakeup()

    def _start_queue_management_thread(self):
        if self._queue_management_thread is None:
            # When the executor gets garbarge collected, the weakref callback
            # will wake up the queue management thread so that it can terminate
            # if there is no pending work item.
            def weakref_cb(_,
                           thread_wakeup=self._queue_management_thread_wakeup):
                mp.util.debug('Executor collected: triggering callback for'
                              ' QueueManager wakeup')
                thread_wakeup.wakeup()
            # Start the processes so that their sentinels are known.
            self._adjust_process_count()
            self._queue_management_thread = threading.Thread(
                target=_queue_management_worker,
                args=(weakref.ref(self, weakref_cb),
                      self._processes,
                      self._pending_work_items,
                      self._work_ids,
                      self._call_queue,
                      self._result_queue,
                      self._queue_management_thread_wakeup),
                name="QueueManagerThread")
            self._queue_management_thread.daemon = True
            self._queue_management_thread.start()
            _threads_wakeups[self._queue_management_thread] = \
                self._queue_management_thread_wakeup

    def _adjust_process_count(self):
        for _ in range(len(self._processes), self._max_workers):
            p = self._mp_context.Process(
                target=_process_worker,
                args=(self._call_queue,
                      self._result_queue,
                      self._initializer,
                      self._initargs))
            p.start()
            self._processes[p.pid] = p

    def submit(*args, **kwargs):
        if len(args) >= 2:
            self, fn, *args = args
        elif not args:
            raise TypeError("descriptor 'submit' of 'ProcessPoolExecutor' object "
                            "needs an argument")
        elif 'fn' in kwargs:
            fn = kwargs.pop('fn')
            self, *args = args
            import warnings
            warnings.warn("Passing 'fn' as keyword argument is deprecated",
                          DeprecationWarning, stacklevel=2)
        else:
            raise TypeError('submit expected at least 1 positional argument, '
                            'got %d' % (len(args)-1))

        with self._shutdown_lock:
            if self._broken:
                raise BrokenProcessPool(self._broken)
            if self._shutdown_thread:
                raise RuntimeError('cannot schedule new futures after shutdown')
            if _global_shutdown:
                raise RuntimeError('cannot schedule new futures after '
                                   'interpreter shutdown')

            f = _base.Future()
            w = _WorkItem(f, fn, args, kwargs)

            self._pending_work_items[self._queue_count] = w
            self._work_ids.put(self._queue_count)
            self._queue_count += 1
            # Wake up queue management thread
            self._queue_management_thread_wakeup.wakeup()

            self._start_queue_management_thread()
            return f
    submit.__text_signature__ = _base.Executor.submit.__text_signature__
    submit.__doc__ = _base.Executor.submit.__doc__

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the process pool.
                If set to one, the items in the list will be sent one at a time.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")

        results = super().map(partial(_process_chunk, fn),
                              _get_chunks(*iterables, chunksize=chunksize),
                              timeout=timeout)
        return _chain_from_iterable_of_lists(results)

    def shutdown(self, wait=True):
        with self._shutdown_lock:
            self._shutdown_thread = True
        if self._queue_management_thread:
            # Wake up queue management thread
            self._queue_management_thread_wakeup.wakeup()
            if wait:
                self._queue_management_thread.join()
        # To reduce the risk of opening too many files, remove references to
        # objects that use file descriptors.
        self._queue_management_thread = None
        if self._call_queue is not None:
            self._call_queue.close()
            if wait:
                self._call_queue.join_thread()
            self._call_queue = None
        self._result_queue = None
        self._processes = None

        if self._queue_management_thread_wakeup:
            self._queue_management_thread_wakeup.close()
            self._queue_management_thread_wakeup = None

    shutdown.__doc__ = _base.Executor.shutdown.__doc__

atexit.register(_python_exit)

# endregion
# endregion


# region Curation
class Curation:
    @staticmethod
    def min_spikes_per_unit(sorting):
        """
        Curate units in sorting.unit_ids based on minimum number of spikes per unit
        Units with greater spikes than min_spikes_per_unit are curated

        Parameters
        ----------
        sorting: KilosortSortingExtractor
            Must have sorting.unit_ids and sorting.get_unit_spike_train(unit_id=)
        Returns
        -------
        Curated unit ids as np.array
        """
        print_stage("CURATING MIN SPIKES PER UNIT")
        stopwatch = Stopwatch()
        curated_unit_ids = []
        for unit_id in sorting.unit_ids:
            num_spikes = sorting.get_unit_spike_train(unit_id=unit_id).size
            if num_spikes > min_spikes_per_unit:
                curated_unit_ids.append(unit_id)
        print(f'N units after minimum spikes per unit curation: {len(curated_unit_ids)}')
        stopwatch.log_time()
        return np.asarray(curated_unit_ids)

    @staticmethod
    def firing_rate(recording, sorting):
        """
        Curate units in sorting.unit_ids based on firing rate
        Units with firing rate greater than fr_thresh are curated

        Parameters
        ----------
        recording
            Must have recording.get_num_samples() and recording.get_sampling_frequency()
        sorting: KilosortSortingExtractor
            Must have sorting.unit_ids and sorting.get_unit_spike_train(unit_id=)

        Returns
        -------
        Curated unit ids as np.array
        """
        print_stage("CURATING FIRING RATE")
        stopwatch = Stopwatch()
        total_duration = recording.get_num_samples() / recording.get_sampling_frequency()
        curated_unit_ids = []
        for unit_id in sorting.unit_ids:
            firing_rate = sorting.get_unit_spike_train(unit_id=unit_id).size / total_duration
            if firing_rate >= fr_thresh:
                curated_unit_ids.append(unit_id)
        print(f'N units after firing rate curation: {len(curated_unit_ids)}')
        stopwatch.log_time()
        return np.asarray(curated_unit_ids)

    @staticmethod
    def isi_violation(recording, sorting, isi_threshold_ms=1.5, min_isi_ms=0):
        """
        Calculate Inter-Spike Interval (ISI) violations for a spike train.

        It computes several metrics related to isi violations:
            * isi_violations_ratio: the relative firing rate of the hypothetical neurons that are generating the ISI
                                    violations. Described in [1]. See Notes.
            * isi_violation_rate: number of ISI violations divided by total rate
            * isi_violation_count: number of ISI violations

        [1] Hill et al. (2011) J Neurosci 31: 8699-8705

        Parameters
        ----------
        recording
            Must have recording.get_num_samples() and recording.get_sampling_frequency()
        sorting
            Must have sorting.unit_ids and sorting.get_unit_spike_train(unit_id=)
        isi_threshold_ms : float
            Threshold for classifying adjacent spikes as an ISI violation. This is the biophysical refractory period
            (default=1.5)
        min_isi_ms : float
            Minimum possible inter-spike interval (default=0). This is the artificial refractory period enforced
            by the data acquisition system or post-processing algorithms

        Returns
        -------
        Curated unit ids as np.array

        Notes
        -----
        isi_violations_ratio : float
            The isi violation ratio described in [1] (violations_rate / total_rate)
        isi_violations_rate : float
            Rate of contaminating spikes as a fraction of overall rate. Higher values indicate more contamination
        isi_violation_count : int
            Number of violations
        -----
        You can interpret an ISI violations ratio value of 0.5 as meaning that contaminating spikes are occurring at roughly
        half the rate of "true" spikes for that unit. In cases of highly contaminated units, the ISI violations value can
        sometimes be even greater than 1.

        Originally written in Matlab by Nick Steinmetz (https://github.com/cortex-lab/sortingQuality) and
        converted to Python by Daniel Denman.
        """
        print_stage("CURATING ISI VIOLATION")
        stopwatch = Stopwatch()
        sampling_frequency = recording.get_sampling_frequency()
        total_duration = recording.get_num_samples() / sampling_frequency

        # All units converted to seconds
        isi_threshold_s = isi_threshold_ms / 1000
        min_isi_s = min_isi_ms / 1000
        isi_threshold_samples = int(isi_threshold_s * sampling_frequency)

        curated_unit_ids = []
        for unit_id in sorting.unit_ids:
            spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
            num_spikes = spike_train.size
            isis = np.diff(spike_train)

            violation_num = np.sum(isis < isi_threshold_samples)
            violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s)

            total_rate = num_spikes / total_duration
            violation_rate = violation_num / violation_time
            violation_rate_ratio = violation_rate / total_rate

            if violation_rate_ratio <= isi_viol_thresh:
                curated_unit_ids.append(unit_id)
        print(f'N units after ISI violation curation: {len(curated_unit_ids)}')
        stopwatch.log_time()
        return np.asarray(curated_unit_ids)

    @staticmethod
    def max_norm_std(waveform_extractor, peak_sign='neg'):
        """
        Curate units in waveform_extractor.sorting.unit_ids based on maximum normalized standard deviation
        Maximum normalized standard deviation is based on
        standard deviation at peak or over entire waveform window (determined by user parameters)
        divided (normalized) by amplitude since higher amplitude units will have greater absolute std

        Units with maximum normalized standard deviation below max_norm_std are curated

        Parameters
        ----------
        waveform_extractor: WaveformExtractor
        peak_sign: str
            Sign of the template to compute amplitude and best channels ('neg', 'pos', 'both')

        Returns
        -------
        Curated unit ids as np.array
        """
        print_stage("CURATING MAX NORMALIZED STD")
        stopwatch = Stopwatch()
        template_amplitudes, channel_max_indices = waveform_extractor.get_template_extremums(peak_sign=peak_sign)
        curated_unit_ids = []
        for unit_id in waveform_extractor.sorting.unit_ids:
            waveforms = waveform_extractor.get_waveforms(unit_id)[:, :, channel_max_indices[unit_id]]
            peak_ind = WaveformExtractor.get_peak_ind(waveforms, peak_sign=peak_sign)
            if max_norm_at_peak:
                std = np.std(waveforms[np.arange(peak_ind.size), peak_ind])
            else:
                nbefore = waveform_extractor.ms_to_samples(max_norm_over_window_ms_before)
                nafter = waveform_extractor.ms_to_samples(max_norm_over_window_ms_after)
                std_sum = 0
                for i, peak_ind in enumerate(peak_ind):
                    std_sum += np.std(waveforms[i, peak_ind-nbefore:peak_ind+nafter])
                std = std_sum / peak_ind.size

            amp_avg = template_amplitudes[unit_id]
            std_scaled = std / amp_avg
            if std_scaled <= max_norm_std:
                curated_unit_ids.append(unit_id)

        print(f'N units after max normalized std: {len(curated_unit_ids)}')
        stopwatch.log_time()
        return np.asarray(curated_unit_ids)

    @staticmethod
    def snr(waveform_extractor, peak_sign='neg'):
        """
        Curate units in waveform_extractor.sorting.unit_ids based on
        ratio of peak amplitude to noise of the channel that defines the peak amplitude

        Parameters
        ----------
        waveform_extractor: WaveformExtractor
        peak_sign: Union('neg', 'pos', 'both')
            Sign of the units' templates to compute best channel

        Returns
        -------
        Curated unit ids as np.array
        """
        print_stage("CURATING SIGNAL-TO-NOISE RATIO")
        stopwatch = Stopwatch()
        recording = waveform_extractor.recording
        sorting = waveform_extractor.sorting

        extremum_amplitudes, extremum_channels_indices = waveform_extractor.get_template_extremums(peak_sign=peak_sign)
        noise_levels = Curation.get_noise_levels(recording, waveform_extractor.return_scaled)
        curated_unit_ids = []
        for unit_id in sorting.unit_ids:
            chan_ind = extremum_channels_indices[unit_id]
            amplitude = extremum_amplitudes[unit_id]
            noise = noise_levels[chan_ind]
            with np.errstate(divide='ignore'):  # Ignore divide by zero warnings
                if amplitude / noise >= snr_thresh:
                    curated_unit_ids.append(unit_id)

        print(f"N units after SNR curation: {len(curated_unit_ids)}")
        stopwatch.log_time()
        return np.asarray(curated_unit_ids)

    # region For getting noise levels for curate_snr
    @staticmethod
    def get_noise_levels(recording, return_scaled=True, **random_chunk_kwargs):
        """
        Estimate noise for each channel using MAD methods.

        Internally, it samples some chunk across segment.
        And then, it uses MAD estimator (more robust than STD)
        """
        stopwatch = Stopwatch()
        print("Getting noise levels per channel")

        random_chunks = Curation.get_random_data_chunks(recording, return_scaled=return_scaled,
                                                        **random_chunk_kwargs)
        med = np.median(random_chunks, axis=0, keepdims=True)
        noise_levels = np.median(np.abs(random_chunks - med), axis=0) / 0.6745
        stopwatch.log_time("Done getting noise levels.")
        return noise_levels

    @staticmethod
    def get_random_data_chunks(recording, return_scaled=False, num_chunks=20, chunk_size=10000, seed=0):
        """
        Extract random chunks from recording

        This is used in get_noise_levels() to estimate noise on traces.

        Parameters
        ----------
        recording: BaseRecording
            The recording to get random chunks from
        return_scaled: bool
            If True, returned chunks are scaled to uV
        num_chunks: int
            Number of chunks
        chunk_size: int
            Size of a chunk in number of frames
        seed: int
            Random seed

        Returns
        -------
        chunk_list: np.array
            Array of chunks
        """

        length = recording.get_num_samples()
        random_state = np.random.RandomState(seed=seed)
        random_starts = random_state.randint(0, length - chunk_size, size=num_chunks)
        chunks = []
        for start_frame in random_starts:
            chunk = recording.get_traces(start_frame=start_frame,
                                         end_frame=start_frame + chunk_size,
                                         return_scaled=return_scaled)
            chunks.append(chunk)
        return np.concatenate(chunks, axis=0)
    # endregion
# endregion


# region Utilities
class Utils:
    """
    Utility functions implemented by spikeinterface
    """

    @staticmethod
    def read_python(path):
        """Parses python scripts in a dictionary

        Parameters
        ----------
        path: str or Path
            Path to file to parse

        Returns
        -------
        metadata:
            dictionary containing parsed file

        """
        from six import exec_
        import re
        path = Path(path).absolute()
        assert path.is_file()
        with path.open('r') as f:
            contents = f.read()
        contents = re.sub(r'range\(([\d,]*)\)', r'list(range(\1))', contents)
        metadata = {}
        exec_(contents, {}, metadata)
        metadata = {k.lower(): v for (k, v) in metadata.items()}
        return metadata

    @staticmethod
    def ensure_n_jobs(recording, n_jobs=1):
        # Ensures that the number of jobs specified is possible by the operating system

        import joblib

        if n_jobs == -1:
            n_jobs = joblib.cpu_count()
        elif n_jobs == 0:
            n_jobs = 1
        elif n_jobs is None:
            n_jobs = 1

        version = sys.version_info

        if (n_jobs != 1) and not (version.major >= 3 and version.minor >= 7):
            print(f"Python {sys.version} does not support parallel processing")
            n_jobs = 1

        if not recording.is_dumpable:
            if n_jobs > 1:
                n_jobs = 1
                print("RecordingExtractor is not dumpable and can't be processed in parallel")

        return n_jobs

    @staticmethod
    def ensure_chunk_size(recording, total_memory=None, chunk_size=None, chunk_memory=None, n_jobs=1, **other_kwargs):
        """
        'chunk_size' is the traces.shape[0] for each worker.

        Flexible chunk_size setter with 3 ways:
            * "chunk_size": is the length in sample for each chunk independently of channel count and dtype.
            * "chunk_memory": total memory per chunk per worker
            * "total_memory": total memory over all workers.

        If chunk_size/chunk_memory/total_memory are all None then there is no chunk computing
        and the full trace is retrieved at once.

        Parameters
        ----------
        chunk_size: int or None
            size for one chunk per job
        chunk_memory: str or None
            must endswith 'k', 'M' or 'G'
        total_memory: str or None
            must endswith 'k', 'M' or 'G'
        """

        if chunk_size is not None:
            # manual setting
            chunk_size = int(chunk_size)
        elif chunk_memory is not None:
            assert total_memory is None
            # set by memory per worker size
            chunk_memory = Utils._mem_to_int(chunk_memory)
            n_bytes = np.dtype(recording.get_dtype()).itemsize
            num_channels = recording.get_num_channels()
            chunk_size = int(chunk_memory / (num_channels * n_bytes))
        if total_memory is not None:
            # clip by total memory size
            n_jobs = Utils.ensure_n_jobs(recording, n_jobs=n_jobs)
            total_memory = Utils._mem_to_int(total_memory)
            n_bytes = np.dtype(recording.get_dtype()).itemsize
            num_channels = recording.get_num_channels()
            chunk_size = int(total_memory / (num_channels * n_bytes * n_jobs))
        else:
            if n_jobs == 1:
                # not chunk computing
                chunk_size = None
            else:
                raise ValueError('For n_jobs >1 you must specify total_memory or chunk_size or chunk_memory')

        return chunk_size

    @staticmethod
    def _mem_to_int(mem):
        # Converts specified memory (e.g. 4G) to integer number
        _exponents = {'k': 1e3, 'M': 1e6, 'G': 1e9}

        suffix = mem[-1]
        assert suffix in _exponents
        mem = int(float(mem[:-1]) * _exponents[suffix])
        return mem
# endregion


# region spikesort_matlab.py
class Stopwatch:
    def __init__(self):
        self._time_start = time.time()

    def log_time(self, text=None):
        if text is None:
            print(f"Time: {time.time() - self._time_start:.2f}s")
        else:
            print(f"{text} Time: {time.time() - self._time_start:.2f}s")


def create_folder(path, parents=True):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=parents)
        print(f"Created folder: {path}")


def load_recording(rec_path):
    print_stage("LOADING MAXWELL RECORDING")
    print(f"Recording path: {rec_path}")
    try:
        stopwatch = Stopwatch()
        if str(rec_path).endswith(".raw.h5"):
            rec = MaxwellRecordingExtractor(rec_path)
        else:
            rec = concatenate_recordings(rec_path)
        assert rec.get_num_segments() == 1, "Recording has multiple segments. Divide recording into multiple recordings"
        stopwatch.log_time("Done loading recording.")
        return rec
    except Exception as e:
        print(f"Could not open the provided file: {rec_path} with the MaxwellRecordingExtractor because of {e}")
        print("Moving on to next recording")
        return -1


def concatenate_recordings(rec_path):
    print("Concatenating recordings")
    recordings = []

    recording_names = natsorted([p.name for p in rec_path.iterdir() if p.name.endswith(".raw.h5")])
    for rec_name in recording_names:
        rec_file = [p for p in rec_path.iterdir() if p.name == rec_name][0]
        rec = MaxwellRecordingExtractor(rec_file)
        recordings.append(rec)
        print(f"{rec_name}: DURATION: {rec.get_num_frames() / rec.get_sampling_frequency()} s -- "
              f"NUM. CHANNELS: {rec.get_num_channels()}")
    rec = si_segmentutils.concatenate_recordings(recordings)
    print(f"Done concatenating {len(recordings)} recordings")
    return rec


def get_paths(rec_path, inter_path):

    rec_path = Path(rec_path)
    print_stage("PROCESSING RECORDING")
    print(f"Recording path: {rec_path}")

    inter_path = Path(inter_path)

    recording_dat_path = inter_path / (rec_path.name.split(".")[0] + "_filtered.dat")
    output_folder = inter_path / 'kilosort2_results'
    waveforms_folder = inter_path / 'waveforms'
    waveforms_raw_folder = waveforms_folder / 'waveforms_raw'
    waveforms_curated_folder = waveforms_folder / 'waveforms_curated'

    delete_folders = []
    if recompute_recording:
        delete_folders.append(inter_path)
    if recompute_sorting:
        delete_folders.extend((output_folder, waveforms_folder))
    if recompute_waveforms:
        delete_folders.append(waveforms_folder)
    if recompute_curation:
        delete_folders.append(waveforms_curated_folder)
    for folder in delete_folders:
        if folder.exists():
            shutil.rmtree(folder)
            print(f"Deleted folder: {folder}")

    create_folder(inter_path)
    return rec_path, inter_path, recording_dat_path, output_folder, waveforms_raw_folder, waveforms_curated_folder


def write_recording(recording, recording_dat_path, verbose=True):
    print_stage("WRITING BINARY RECORDING")
    stopwatch = Stopwatch()

    print("Using bandpass filter")
    recording_filtered = bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max)

    print(f"Kilosort2's .dat path: {recording_dat_path}")
    if recompute_recording or not recording_dat_path.exists():
        # dtype has to be 'int16' (that's what Kilosort2 expects)
        # BinaryRecordingExtractor.write_recording(recording_filtered, file_paths=recording_dat_path,
        #                                          dtype='int16', total_memory=total_memory,
        #                                          n_jobs=n_jobs, verbose=verbose, progress_bar=verbose)
        BinaryRecordingExtractor.write_recording(recording_filtered, file_paths=recording_dat_path,
                                                 dtype='int16', chunk_memory=chunk_memory,
                                                 n_jobs=n_jobs, verbose=verbose, progress_bar=verbose)
    else:
        print(f"Skipping writing recording.dat\nUsing existing {recording_dat_path} as recording file")
    stopwatch.log_time("Done writing recording.")
    return recording_filtered


def spike_sort(rec_cache, rec_path, recording_dat_path, output_folder):
    print_stage("SORTING")
    stopwatch = Stopwatch()
    sorting = -1

    kilosort = RunKilosort()
    try:
        if not recompute_sorting and (output_folder / "spike_times.npy").exists():
            print('Skipping', rec_path, ' since already sorted')
            sorting = kilosort.get_result_from_folder(output_folder)
        else:
            create_folder(output_folder)
            sorting = kilosort.run(recording=rec_cache, recording_dat_path=recording_dat_path, output_folder=output_folder)

    except Exception as e:
        print(f"Kilosort2 failed on recording {rec_path} because of {e}")

    stopwatch.log_time("Done sorting.")

    return sorting


def extract_waveforms(recording, sorting, folder,
                      **job_kwargs):
    """
    Extracts waveform on paired Recording-Sorting objects.
    Waveforms are persistent on disk and cached in memory.

    Parameters
    ----------
    recording: Recording
        The recording object
    sorting: Sorting
        The sorting object
    folder: str or Path
        The folder where waveforms are cached

    Returns
    -------
    we: WaveformExtractor
        The WaveformExtractor object that represents the waveforms
    """

    print_stage("EXTRACTING WAVEFORMS")
    stopwatch = Stopwatch()

    if not recompute_waveforms and (folder/"waveforms").is_dir():  # Load saved waveform extractor
        print("Loading waveforms from folder")
        we = WaveformExtractor.load_from_folder(recording, sorting, folder)
        stopwatch.log_time("Done extracting waveforms.")
    else:  # Create new waveform extractor
        print("Computing new waveforms")
        create_folder(folder)
        we = WaveformExtractor.create(recording, sorting, folder)
        we.run_extract_waveforms(**job_kwargs)
        we.compute_templates(modes=('average',))  # add 'std'?
        stopwatch.log_time("Done extracting waveforms.")
    return we


def curate_units(recording, sorting, we_raw, curated_folder):
    print_stage("CURATING UNITS")
    stopwatch = Stopwatch()
    if not recompute_curation and (curated_folder/'waveforms').exists():
        print("Skipping data curation since already curated.")
        print("Loading saved curated data.")
        we_curated = WaveformExtractor.load_from_folder(recording, sorting, curated_folder)
        stopwatch.log_time("Done loading.")
        return we_curated

    if not auto_curate:
        print("'auto_curate' is set to False, so skipping data curation.")
        return we_raw

    # Perform curation
    unit_ids_initial = sorting.unit_ids
    print(f'N units before curation: {len(unit_ids_initial)}')

    # Minimum spikes per unit
    if min_spikes_per_unit is not None:
        unit_ids_curated_min_spikes = Curation.min_spikes_per_unit(sorting)
        sorting.unit_ids = unit_ids_curated_min_spikes

    # Firing rate
    if fr_thresh is not None:
        unit_ids_curated_fr = Curation.firing_rate(recording, sorting)
        sorting.unit_ids = unit_ids_curated_fr

    # Interspike interval
    if isi_viol_thresh is not None:
        unit_ids_curated_isi = Curation.isi_violation(recording, sorting)
        sorting.unit_ids = unit_ids_curated_isi

    # Maximum normalized standard deviation
    if max_norm_std is not None:
        unit_ids_curated_max_std = Curation.max_norm_std(we_raw)
        sorting.unit_ids = unit_ids_curated_max_std

    # Signal-to-noise ratio
    if snr_thresh is not None:
        unit_ids_curated_snr = Curation.snr(we_raw)
        sorting.unit_ids = unit_ids_curated_snr

    # Save only the curated waveforms to a new file
    we_curated = we_raw.save_curated_units(unit_ids=sorting.unit_ids, curated_folder=curated_folder)

    return we_curated


def process_recording(rec_path, inter_path):
    # Start time for entire process
    stopwatch = Stopwatch()

    # Get Paths
    rec_path, inter_path, recording_dat_path, output_folder, waveforms_raw_folder, waveforms_curated_folder = get_paths(rec_path, inter_path)

    # Save a copy of the script
    if save_script:
        print_stage("SAVING SCRIPT")
        copy_script(inter_path)

    # Load Recording
    recording_raw = load_recording(rec_path)
    if recording_raw == -1:  # Could not load recording
        return "Could not load recording. Moving on to next recording."

    # Filter and Dumping
    recording_filtered = write_recording(recording_raw, recording_dat_path, verbose=True)

    # Spike Sorting
    sorting = spike_sort(rec_cache=recording_filtered, rec_path=rec_path, recording_dat_path=recording_dat_path,
                         output_folder=output_folder)
    if sorting == -1:  # Could not sort recording
        return "Could not sort recording. Moving on to next recording"

    # Extract waveforms
    we_raw = extract_waveforms(recording_filtered, sorting, folder=waveforms_raw_folder,
                               n_jobs=n_jobs, total_memory=total_memory, progress_bar=True)

    # Curating Data
    waveform_extractor = curate_units(recording_filtered, sorting, we_raw, waveforms_curated_folder)

    stopwatch.log_time("\nDone processing recording.")

    return waveform_extractor


def convert_to_matlab(waveform_extractor, rec_path, matlab_path):
    print_stage("CONVERTING TO MATLAB")
    stopwatch = Stopwatch()

    # Get matlab file path
    name = str(rec_path).split(r"/")[-1].split(".")[0] + "_sorted.mat"
    mat_file = Path(matlab_path) / name

    if mat_file.exists():
        if override_matlab:
            mat_file.unlink()
            print(f"Deleted {mat_file}")
        else:
            print(f"Skipping creating {mat_file} since already created")
            return

    # Make parent file
    create_folder(mat_file.parent)

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = waveform_extractor.load_unit_ids()
    sorting.unit_ids = unit_ids  # sorting.unit_ids may not be updated to curated unit_ids if recompute_curation=False
    mdict = {"units": [], "locations": recording.get_channel_locations(), "fs": recording.get_sampling_frequency()}

    # Get max channels
    _, extremum_channels = waveform_extractor.get_template_extremums()

    # Get channel locations
    locations = recording.get_channel_locations()

    # Get electrodes
    if save_electrodes:
        electrodes = recording.get_property('electrode')

    center_spike_times_on_peak_corrected = center_spike_times_on_peak
    if center_spike_times_on_peak and max_waveforms_per_unit is not None:
        print("Setting 'center_spike_times_on_peak' to False because 'max_waveforms_per_unit' is not None")
        # To allow 'center_spike_times_on_peak' to be True and 'max_waveforms_per_unit' to be None,
        # the traces of the recording must be retrieved WHICH IS SLOW and memory intensive
        # The current method uses the extracted waveforms as reference to recenter the peaks
        center_spike_times_on_peak_corrected = False
    if center_spike_times_on_peak_corrected:
        time_centering_spike_train = 0

    for u in unit_ids:
        unit_dict = {}
        max_channel_ind = extremum_channels[u]
        max_location = locations[max_channel_ind]
        template = waveform_extractor.get_template(u)

        spike_train = sorting.get_unit_spike_train(u)
        if center_spike_times_on_peak_corrected:
            start_time = time.time()
            waveforms = waveform_extractor.get_waveforms(u)[:, :, max_channel_ind]
            peak_ind = waveform_extractor.get_peak_ind(waveforms)
            spike_train_offsets = peak_ind - waveforms.shape[1]//2
            if save_spike_train_offsets:
                unit_dict["spike_train_offsets"] = spike_train_offsets
            spike_train = spike_train - spike_train_offsets
            time_centering_spike_train += time.time() - start_time

        unit_dict["spike_train"] = spike_train
        unit_dict["x_max"] = max_location[0]
        unit_dict["y_max"] = max_location[1]
        unit_dict["template"] = template
        unit_dict["unit_id"] = u

        if save_electrodes:
            electrode = electrodes[max_channel_ind]
            unit_dict["electrode"] = electrode

        mdict["units"].append(unit_dict)

    if center_spike_times_on_peak_corrected:
        print(f"Time to center all spikes on negative peak: {time_centering_spike_train:.2f}s")
    savemat(mat_file, mdict)
    print(f"Saved to {mat_file}")
    stopwatch.log_time()


def copy_script(path):
    # Copies script and stores it in path
    # https://stackoverflow.com/questions/23321100/best-way-to-have-a-python-script-copy-itself/49210778
    copied_script_name = time.strftime("%Y-%m-%d_%H%M") + '_' + os.path.basename(__file__)
    copied_path = (path / copied_script_name).absolute()
    shutil.copyfile(__file__, copied_path)
    print(f"Saved a copy of script to {copied_path}")


def main():
    for (rec_path, inter_path, matlab_path) in zip(recording_files, intermediate_folders, matlab_folders):
        stopwatch = Stopwatch()
        result = process_recording(rec_path, inter_path)
        if type(result) == str:
            print(result)
            continue

        convert_to_matlab(result, rec_path, matlab_path)

        print_stage(f"DONE WITH RECORDING")
        print(f"Recording: {rec_path}")
        stopwatch.log_time("Total")
        print("\n\n")


if __name__ == "__main__":
    main()
# endregion
