"""
Template script to spike sort a list of recordings using SpikeInterface>=0.90.

Original Author: Alessio Buccino
"""

######################################################
################  FOLDER LOCATIONS ###################
######################################################

# List of data files you want to spike sort
recording_files = [
    "/home/maxlim/SpikeSorting/data/maxone/200123/2953/recordings",
    # "/home/maxlim/SpikeSorting/data/maxone/200123/2954/network/10h36m24s.raw.h5",
    # "/home/maxlim/SpikeSorting/data/maxone/200123/2957/network/11h24m49s.raw.h5",
    # "/home/maxlim/SpikeSorting/data/maxone/200520/5116/network/11h54m18s.raw.h5",
    # "/home/maxlim/SpikeSorting/data/maxone/200520/5118/network/12h09m06s.raw.h5"
]

# List of intermediate folders where tmp and output files are saved
intermediate_folders = [
    "/home/maxlim/SpikeSorting/data/maxone/200123/2953/processed/max",
    # "/home/maxlim/SpikeSorting/data/maxone/200123/2954/processed",
    # "/home/maxlim/SpikeSorting/data/maxone/200123/2957/processed",
    # "/home/maxlim/SpikeSorting/data/maxone/200520/5116/processed",
    # "/home/maxlim/SpikeSorting/data/maxone/200520/5118/processed"
]

# List of .mat paths where final matlab files are saved.
matlab_folders = [
    "/home/maxlim/SpikeSorting/data/maxone/200123/2953/sorted/max/new_filtered_2953_sorted.mat",
    # "/home/maxlim/SpikeSorting/data/maxone/200123/2954/sorted",
    # "/home/maxlim/SpikeSorting/data/maxone/200123/2957/sorted",
    # "/home/maxlim/SpikeSorting/data/maxone/200520/5116/sorted",
    # "/home/maxlim/SpikeSorting/data/maxone/200520/5118/sorted",
]

assert len(recording_files) == len(intermediate_folders) == len(matlab_folders), "'recording_files' and 'intermediate_folders' and 'MATLAB_FOLDERS' " \
                                                                                 "should all have the same length"

######################################################
###############  KILOSORT PARAMETERS  ################
######################################################

# Set toolbox paths
kilosort_path = "/home/maxlim/SpikeSorting/Kilosort2"  # "E:/KosikLab/SpikeSorting/Kilosort2"
hdf5_plugin_path = '/home/maxlim/SpikeSorting/extra_libraries/'  # "E:/KosikLab/SpikeSorting/extra_libraries/"

# Sorter params
kilosort_params = {
    'detect_threshold': 6,
    'projection_threshold': [10, 4],
    'preclust_threshold': 8,
    'car': True,
    'minFR': 0.1,
    'minfr_goodchannels': 0.1,
    'FREQ_MIN': 150,
    'sigmaMask': 30,
    'nPCs': 3,
    'ntbuff': 64,
    'nfilt_factor': 4,
    'NT': None,  # 65600
    'keep_good_only': False,
    'TOTAL_MEMORY': "24G",  # Too low memory will cause error
    'n_jobs_bin': 1
}

######################################################
###############  RECOMPUTE RESULTS?  #################
######################################################

# If True and is present, it's deleted and recomputed
rewrite_recording = True  # Refers to the .dat recording file created for Kilosort
recompute_sorting = True
recompute_curation = True
recompute_waveforms = True

# Override matlab file if it exists
override_matlab = True

######################################################
############  BANDPASS FILTER PARAMETERS  ############
######################################################

freq_min = 300
freq_max = 6000

######################################################
###############  CURATION PARAMETERS  ################
######################################################

# If True, output is automatically curated using quality metrics (QC)
auto_curate = True

# ms before waveform trough to include
ms_before = 3.
# ms after waveform trough to include
ms_after = 3.  # spikesort_drug_experiment uses 3 while original (new API) uses 4.
# If True and recording has gain_to_uV/offset_to_uV properties, waveforms are converted to uV
return_scaled = True

# Thresholds for automatic curations (if any is None, that autocuration is skipped
# ISI-violation rate ratio (greater values are removed)
#   Ratio of violation_rate / total_spike_rate
isi_viol_thresh = 0.3
# firing rate (smaller values are removed)
fr_thresh = 0.05
# signal-to-noise ratio (smaller values are removed)
snr_thresh = 5

# Number of jobs to use for curation
# DEFAULT: Same as Kilosort
n_jobs = kilosort_params["n_jobs_bin"]
# Total RAM to use for curation
# DEFAULT: Same as Kilosort
total_memory = kilosort_params["TOTAL_MEMORY"]

# Number of spikes per unit to compute templates (None-> all spikes are used)
max_spikes_per_unit = 500

######################################################
################  MATLAB PARAMETERS  #################
######################################################
# If True, include the electrode number [1, 26400] to the "unit" data structure
save_electrodes = True

######################################################
######################  CODE  ########################
######################################################

# region Import Modules
# print("Importing packages")
import spikeinterface.extractors as se
import spikeinterface.toolkit as st

import numpy as np
import matplotlib as mpl

mpl.use("agg")

import json
from spikeinterface.extractors import BinaryRecordingExtractor
from spikeinterface.core.core_tools import check_json
import spikeinterface.core.segmentutils as si_segmentutils

import datetime

import subprocess
import tempfile
import shutil
import signal
from pathlib import Path
import time
from typing import Optional, List, Any, Union
from tqdm import tqdm
from collections import namedtuple
from scipy.io import savemat
import pandas as pd
from natsort import natsorted

# endregion


# region Kilosort
class RunKilosort:
    # Must create Kilosort instance to 1) set paths 2) check if kilosort is installed
    def __init__(self):
        # Set paths
        self.path = self.set_kilosort_path(kilosort_path)  # "/home/maxlim/Kilosort2"
        os.environ['HDF5_PLUGIN_PATH'] = hdf5_plugin_path  # '/home/maxlim/SpikeSorting/extra_libraries/'

        # Check if kilosort is installed
        if not self.check_if_installed():
            raise Exception(f"Kilosort is not installed.")

        # Make sure parameters are formatted correctly
        RunKilosort.format_params()

    # Run kilosort
    def run(self, recording, recording_dat_path, output_folder, verbose=False, raise_error=True):

        # region REMOVED
        # ORIGINAL PURPOSE: Check if kilosort can process recording
        # REASON FOR REMOVAL: Being checked earlier now
        # STEP 0) Check if kilosort can process recording
        # if recording.get_num_segments() > 1:
        #     raise ValueError(
        #         f'Kilosort can not handle multi segment, manually concatenate recordings or use spikeinterface\'s concatenate_recordings(...)')
        # endregion

        # STEP 1) Creates kilosort and recording files needed to run kilosort
        self.setup_recording_files(recording, recording_dat_path, output_folder)

        # STEP 2) Actually run kilosort
        self.start_sorting(output_folder, raise_error, verbose)

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
            save('rez.mat', 'rez')
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
        ops.fshigh = {FREQ_MIN};

        % minimum firing rate on a "good" channel (0 to skip)
        ops.minfr_goodchannels = {minfr_goodchannels};

        % threshold on projections (like in Kilosort1, can be different for last pass like [10 4])
        ops.Th = {projection_threshold};

        % how important is the amplitude penalty (like in Kilosort1, 0 means not used, 10 is average, 50 is a lot)
        ops.lam = 10;

        % splitting a cluster at the end requires at least this much isolation for each sub-cluster (mx = 1)
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
        % ops.Nfilt             = 1024; % mx number of clusters
        ops.nfilt_factor        = {nfilt_factor}; % mx number of clusters per good channel (even temporary ones) 4
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
            dat_file=str(recording_dat_path.absolute()),  # str((output_folder / 'recording.dat').absolute()),
            projection_threshold=kilosort_params['projection_threshold'],
            preclust_threshold=kilosort_params['preclust_threshold'],
            minfr_goodchannels=kilosort_params['minfr_goodchannels'],
            minFR=kilosort_params['minFR'],
            freq_min=kilosort_params['FREQ_MIN'],
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
        # need setup_recording to be done.
        output_folder = Path(output_folder)

        now = datetime.datetime.now()
        log = {
            'sorter_name': 'kilosort2',
            # 'sorter_version': str(SorterClass.get_sorter_version()),
            'datetime': now,
            'runtime_trace': []
        }
        t0 = time.perf_counter()

        try:
            self.execute_kilosort_file(output_folder, verbose)
            t1 = time.perf_counter()
            run_time = float(t1 - t0)
            has_error = False
        except Exception as err:
            has_error = True
            run_time = None
            log['error'] = True
            log['error_trace'] = traceback.format_exc()

        log['error'] = has_error
        log['run_time'] = run_time

        # Kilosort has a log file dir to shellscript launcher
        runtime_trace_path = output_folder / 'kilosort2.log'
        runtime_trace = []
        if runtime_trace_path.is_file():
            with open(runtime_trace_path, 'r') as fp:
                line = fp.readline()
                while line:
                    runtime_trace.append(line.strip())
                    line = fp.readline()
        log['runtime_trace'] = runtime_trace

        # Dump to json
        with (output_folder / 'spikeinterface_log.json').open('w', encoding='utf8') as f:
            json.dump(check_json(log), f, indent=4)

        if verbose:
            if has_error:
                print('Error running kilosort2')
            else:
                print(f'kilosort2 run time: {run_time:0.2f}s')

        if has_error and raise_error:
            print(log['error_trace'])
            raise Exception(
                f"Spike sorting failed. You can inspect the runtime trace in {output_folder}/spikeinterface_log.json")

        return run_time

    @staticmethod
    def execute_kilosort_file(output_folder, verbose):
        print('Running kilosort file')

        if 'win' in sys.platform and sys.platform != 'darwin':
            shell_cmd = f'''
                        cd {output_folder}
                        matlab -nosplash -wait -log -r kilosort2_master
                    '''
        else:
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
            print("Setting KILOSORT_PATH environment variable for subprocess calls to:", path)
            os.environ["KILOSORT_PATH"] = path
        except Exception as e:
            print("Could not set KILOSORT_PATH environment variable:", e)

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
        # region REMOVED
        # ORIGINAL PURPOSE: Checks if spikeinterface_log.json exists and raises error if it doesn't exist or if there is an error reported by the file
        # REASON FOR REMOVAL: If there was an error, it would have been caught by the time this function is called
        # # check errors in log file
        # log_file = output_folder / 'spikeinterface_log.json'
        # if not log_file.is_file():
        #     raise Exception('get result error: the output folder does not contain spikeinterface_log.json')
        #
        # with log_file.open('r', encoding='utf8') as f:
        #     log = json.load(f)
        #
        # if bool(log['error']):
        #     raise Exception(
        #         "Spike sorting failed. You can inspect the runtime trace in spikeinterface_log.json")
        # endregion

        # region REMOVED
        # ORIGINAL PURPOSE: Get the kilosort params from the json file
        # REASON FOR REMOVAL: The kilosort params are stored in this file as a dict, so they do not need to be loaded from a json
        # with (output_folder / 'spikeinterface_params.json').open('r') as f:
        #     sorter_params = json.load(f)['sorter_params']
        # endregion
        sorting = KilosortSortingExtractor(folder_path=output_folder, keep_good_only=kilosort_params["keep_good_only"])

        # region REMOVED
        # ORIGINAL PURPOSE: Loads the original data recording as a spikeinterface object and stores it in the KilosortExtractor object
        # REASON FOR REMOVAL: Not needed
        # recording = load_extractor(output_folder / 'spikeinterface_recording.json')
        # if recording is not None:
        #     # can be None when not dumpable
        #     sorting.register_recording(recording)
        # endregion
        return sorting


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

    def __init__(self, folder_path, exclude_cluster_groups=None, keep_good_only=False):
        # region REMVOED
        # ORIGINAL PURPOSE: Import pandas and check if installed
        # REASON FOR REMOVAL: Added to import statements at top of file
        # try:
        #     import pandas as pd
        #     have_pd = True
        # except ImportError:
        #     have_pd = False
        # if not have_pd:
        #     # Error Message when pandas is not installed
        #     raise ImportError("To perform quality metrics and curate the data from Kilosort, install pandas: \n\n pip install pandas\n\n")
        # endregion

        # Folder containing the numpy results of Kilosort
        phy_folder = Path(folder_path)
        self.folder = phy_folder.absolute()

        spike_times = np.load(str(phy_folder / 'spike_times.npy'))
        spike_clusters = np.load(str(phy_folder / 'spike_clusters.npy'))

        # The unit_ids with at least 1 spike
        unit_ids_with_spike = set(spike_clusters.flatten())

        # region REMOVED
        # ORIGINAL PURPOSE: If spike_clusters.npy does not exist, make spike_clusters equal to spike_templates
        # REASON FOR REMOVAL: spike_clusters.npy should always be created
        # spike_templates = np.load(phy_folder / 'spike_templates.npy')
        # if (phy_folder / 'spike_clusters.npy').is_file():
        #     spike_clusters = np.load(phy_folder / 'spike_clusters.npy')
        # else:
        #     spike_clusters = spike_templates
        # endregion

        spike_times.astype(int)
        params = Utils.read_python(str(phy_folder / 'params.py'))
        self.sampling_frequency = params['sample_rate']

        # region REMOVED
        # ORIGINAL PURPOSE: Try to load cluster info (files with cluster_info in name)
        # REASON FOR REMOVAL: No files will contain cluster_info in the name
        # Try to load cluster info
        # cluster_info_files = [p for p in phy_folder.iterdir() if p.suffix in ['.csv', '.tsv']
        #                       and "cluster_info" in p.name]
        #
        # if len(cluster_info_files) == 1:
        #     # load properties from cluster_info file
        #     cluster_info_file = cluster_info_files[0]
        #     if cluster_info_file.suffix == ".tsv":
        #         delimeter = "\t"
        #     else:
        #         delimeter = ","
        #     cluster_info = pd.read_csv(cluster_info_file, delimiter=delimeter)
        # else:
        #     # load properties from other tsv/csv files
        #     all_property_files = [p for p in phy_folder.iterdir() if p.suffix in ['.csv', '.tsv']]
        #
        #     cluster_info = None
        #     for file in all_property_files:
        #         if file.suffix == ".tsv":
        #             delimeter = "\t"
        #         else:
        #             delimeter = ","
        #         new_property = pd.read_csv(file, delimiter=delimeter)
        #         if cluster_info is None:
        #             cluster_info = new_property
        #         else:
        #             cluster_info = pd.merge(cluster_info, new_property, on='cluster_id')
        # endregion

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

        # region REMOVED
        # ORIGINAL PURPOSE: Use SpikeInterface's IDs
        # REASON FOR REMOVAL: Not using SpikeInterface
        # if 'si_unit_id' in cluster_info.columns:
        #     unit_ids = cluster_info["si_unit_id"].values
        #     del cluster_info["si_unit_id"]
        # else:
        #     unit_ids = cluster_info["cluster_id"].values
        # endregion

        # region REMOVED
        # Original Purpose: Sets properties
        # REASON FOR REMOVAL: HOPING that properties are only for user, not API
        # del cluster_info["cluster_id"]
        # for prop_name in cluster_info.columns:
        #     if prop_name in ['chan_grp', 'ch_group']:
        #         self.set_property(key="group", values=cluster_info[prop_name])
        #     elif prop_name != "group":
        #         self.set_property(key=prop_name, values=cluster_info[prop_name])
        #     elif prop_name == "group":
        #         # rename group property to 'quality'
        #         self.set_property(key="quality", values=cluster_info[prop_name])
        # endregion

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

        # For converting Kilosort2's channel index to recording index (since Kilosort exlcudes some bad channels)
        self.channel_map = np.load(str(phy_folder / "channel_map.npy")).flatten()

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
                             return_times: bool = False
                             ):
        # region segment_index = self._check_segment_index(segment_index)
        if segment_index is None:
            if self.get_num_segments() == 1:
                segment_index = 0
            else:
                raise ValueError()
        # endregion

        segment = self._sorting_segments[segment_index]

        spike_train = segment.get_unit_spike_train(
            unit_id=unit_id, start_frame=start_frame, end_frame=end_frame).astype("int64")

        assert not return_times, "Returning times is not implemented yet"

        return spike_train

        # region REMOVED
        # ORIGINAL PURPOSE: To allow get_unit_spike_train to return times or train
        # REASON FOR REMOVAL: return_times should always be False
        # if return_times:  # Hopefully this stays False
        #     if self.has_recording():
        #         times = self.get_times(segment_index=segment_index)
        #         return times[spike_train]
        #     else:
        #         return spike_train / self.get_sampling_frequency()
        # else:
        #     return spike_train
        # endregion

    def get_templates(self):
        """
        Returns templates of sorted units based on the .npy results from Kilosort2
            The templates will have fewer channels than the recording since
            Kilosort2 excludes useless channels

        Returns
        -------
        Templates as np.array with shape (nTemplates, nTimePoints, nTempChannels)
        """
        all_templates = np.load(str(self.folder / "templates.npy"))
        return all_templates[self.unit_ids, :, :]

    def get_templates_raw(self):
        """
        Returns raw templates from Kilosort2's "rez" structure (rez.mat)
        Wraw: the spike template, un-whitened by the operation Wraw(:,:,n) = Wrotinv' * (U(:,n,:) * W(:,n,:)'),
        for each template n.

        Returns
        -------
        Templates as np.array with shape (nTemplates, nTimePoints, nTempChannels)
        """

        from scipy.io import loadmat
        rez_mat = loadmat(str(self.folder / "rez.mat"))
        rez = rez_mat["rez"]
        wraw = rez["Wraw"][0, 0]
        wraw = wraw.transpose((2, 1, 0))
        return wraw

# endregion


# region Auto Curation
class WaveformExtractor:
    def __init__(self, recording, sorting, folder):
        assert recording.get_num_segments() == sorting.get_num_segments(), \
            "The recording and sorting objects must have the same number of segments!"

        sampling_frequency = recording.get_sampling_frequency()
        np.testing.assert_almost_equal(sampling_frequency, sorting.sampling_frequency, decimal=2)

        # region REMOVED
        # ORIGINAL PURPOSE: Recording that is loaded needs to have been filtered
        # REASON FOR REMOVAL: Recording should already be filtered by now
        # if not recording.is_filtered():
        #     raise Exception('The recording is not filtered, you must filter it using `bandpass_filter()`.'
        #                     'If the recording is already filtered, you can also do `recording.annotate(is_filtered=True)')
        # endregion

        self.recording = recording
        self.sampling_frequency = sampling_frequency

        self.sorting = sorting
        self.folder = Path(folder)

        # Cache in memory
        self._waveforms = {}
        self._template_cache = {}
        self._params = {}

        # region REMOVED
        # Load waveform extractor parameters
        # Only happens if waveform extractor was previously created and is now being loaded
        #      SO it was MOVED to load_from_folder
        # if (self.folder / 'params.json').is_file():
        #     with open(str(self.folder / 'params.json'), 'r') as f:
        #         self._params = json.load(f)
        # endregion

    # region Properties containing metadata needed for extraction
    @property
    def nbefore(self):
        return int(self._params['WAVEFORMS_MS_BEFORE'] * self.sampling_frequency / 1000.)

    @property
    def nafter(self):
        return int(self._params['WAVEFORMS_MS_AFTER'] * self.sampling_frequency / 1000.)

    @property
    def nsamples(self):
        return self.nbefore + self.nafter

    @property
    def return_scaled(self):
        return self._params['RETURN_SCALED']

    # endregion

    # region Loading saved waveform extractor
    @classmethod
    def load_from_folder(cls, recording, sorting, folder):
        # region REMOVED
        # ORIGINAL PURPOSE: Load raw recording and sorting output as spikeinterface object
        # REASON FOR REMOVAL: Should already be loaded by this point
        # folder = Path(folder)
        # assert folder.is_dir(), f'This folder does not exists {folder}'
        # recording = load_extractor(folder / 'recording.json',
        #                            base_folder=folder)
        # sorting = load_extractor(folder / 'sorting.json',
        #                          base_folder=folder)
        # endregion

        we = cls(recording, sorting, folder)

        # region Loads saved parameters from JSON
        # MOVED FROM __init__
        # REASON FOR REMOVAL: params are stored in RAM, not file
        # if (we.folder / 'params.json').is_file():
        #     with open(str(we.folder / 'params.json'), 'r') as f:
        #         we._params = json.load(f)
        # endregion

        _possible_template_modes = ('average', 'std', 'median')
        for mode in _possible_template_modes:
            # Load cached templates
            template_file = folder / f'templates_{mode}.npy'
            if template_file.is_file():
                we._template_cache[mode] = np.load(template_file)

        return we

    # endregion

    # region Creating new waveform extractor
    @classmethod
    def create(cls, recording, sorting, folder):
        folder = Path(folder)

        # region REMOVED
        # ORIGINAL PURPOSE: Stops data being written if folder already exists
        # REASON FOR REMOVAL: This would have been checked by now
        # if folder.is_dir():
        #     if remove_if_exists:
        #         shutil.rmtree(folder)
        #     else:
        #         raise FileExistsError('Folder already exists')
        # endregion

        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'waveforms').mkdir(parents=True)

        # region REMOVED
        # ORIGINAL PURPOSE: Save recording and sorting spikeinterface objects as JSON to be retrieved later
        # REASON FOR REMOVAL: The recording and sorting objects do not need to be retrieved later
        # if recording.is_dumpable:
        #     recording.dump(folder / 'recording.json', relative_to=relative_to)
        # if sorting.is_dumpable:
        #     sorting.dump(folder / 'sorting.json', relative_to=relative_to)
        # endregion

        return cls(recording, sorting, folder)

    def set_params(self, dtype=None):
        """
        Set parameters for waveform extraction using parameters specified at beginning of script
        """

        # region REMOVED
        # ORIGINAL PURPOSE: Reset parameters of the object
        # REASON FOR REMOVAL: Not necessary to reset parameters
        # self._reset()
        # endregion

        if dtype is None:
            dtype = self.recording.get_dtype()

        return_scaled_corrected = return_scaled
        if return_scaled_corrected:
            # Check if recording has scaled values:
            if not self.recording.has_scaled_traces():
                print("Setting 'RETURN_SCALED' to False")
                return_scaled_corrected = False

        if np.issubdtype(dtype, np.integer) and return_scaled_corrected:
            dtype = "float32"

        dtype = np.dtype(dtype)

        self._params = dict(
            ms_before=float(ms_before),
            ms_after=float(ms_after),
            max_spikes_per_unit=max_spikes_per_unit,
            return_scaled=return_scaled_corrected,
            dtype=dtype.str)

        # region Saves parameters to be used later if waveform extractor is saved
        # (self.folder / 'params.json').write_text(
        #     json.dumps(check_json(self._params), indent=4), encoding='utf8')
        # endregion

    # endregion

    # region Extract waveforms
    def run_extract_waveforms(self, **job_kwargs):
        p = self._params
        num_chans = self.recording.get_num_channels()
        nbefore = self.nbefore
        nafter = self.nafter
        return_scaled = self.return_scaled

        job_kwargs["N_JOBS"] = Utils.ensure_n_jobs(self.recording, job_kwargs.get('N_JOBS', None))

        selected_spikes = self.sample_spikes()

        # get spike times
        selected_spike_times = {}
        for unit_id in self.sorting.unit_ids:
            selected_spike_times[unit_id] = []
            for segment_index in range(self.sorting.get_num_segments()):
                spike_times = self.sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                sel = selected_spikes[unit_id][segment_index]
                selected_spike_times[unit_id].append(spike_times[sel])

        # prepare memmap
        wfs_memmap = {}
        for unit_id in self.sorting.unit_ids:
            file_path = self.folder / 'waveforms' / f'waveforms_{unit_id}.npy'
            n_spikes = np.sum([e.size for e in selected_spike_times[unit_id]])
            shape = (n_spikes, self.nsamples, num_chans)
            wfs = np.zeros(shape, dtype=p['dtype'])
            np.save(file_path, wfs)
            # wfs = np.load(file_path, mmap_mode='r+')
            wfs_memmap[unit_id] = file_path

        # and run
        func = WorkerChunk.waveform_extractor_chunk
        init_func = WorkerChunk.init_worker_waveform_extractor
        # region UPDATED
        # ORIGINAL PURPOSE: Saves recording/extractor as a dictionary just to be reverted back to object later
        # REASON FOR REMOVAL: No point to save it if it will be retrieved later - Maybe for memory?
        # if N_JOBS == 1:
        #     init_args = (self.recording, self.sorting,)
        # else:
        #     init_args = (self.recording.to_dict(), self.sorting.to_dict(),)

        init_args = (self.recording, self.sorting,)
        # endregion

        init_args = init_args + (wfs_memmap, selected_spikes, selected_spike_times, nbefore, nafter, return_scaled)
        processor = ChunkRecordingExecutor(self.recording, func, init_func, init_args, job_name='extract waveforms',
                                           **job_kwargs)
        processor.run()

    def sample_spikes(self):
        nbefore = self.nbefore
        nafter = self.nafter

        selected_spikes = WaveformExtractor.select_random_spikes_uniformly(self.recording, self.sorting,
                                                                           self._params['max_spikes_per_unit'], nbefore, nafter)

        # store in a 2 columns (spike_index, segment_index) in a npy file
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
            np.save(sampled_index_file, sampled_index)

        return selected_spikes

    @staticmethod
    def select_random_spikes_uniformly(recording, sorting, max_spikes_per_unit, nbefore=None, nafter=None):
        """
        Uniform random selection of spike across segment per units.

        This function does not select spikes near border if nbefore/nafter are not None.
        """
        unit_ids = sorting.unit_ids
        num_seg = sorting.get_num_segments()

        selected_spikes = {}
        for unit_id in unit_ids:
            # spike per segment
            n_per_segment = [sorting.get_unit_spike_train(unit_id, segment_index=i).size for i in range(num_seg)]
            cum_sum = [0] + np.cumsum(n_per_segment).tolist()
            total = np.sum(n_per_segment)
            if max_spikes_per_unit is not None:
                if total > max_spikes_per_unit:
                    global_inds = np.random.choice(total, size=max_spikes_per_unit, replace=False)
                    global_inds = np.sort(global_inds)
                else:
                    global_inds = np.arange(total)
            else:
                global_inds = np.arange(total)
            sel_spikes = []
            for segment_index in range(num_seg):
                in_segment = (global_inds >= cum_sum[segment_index]) & (global_inds < cum_sum[segment_index + 1])
                inds = global_inds[in_segment] - cum_sum[segment_index]

                if max_spikes_per_unit is not None:
                    # clean border when sub selection
                    assert nafter is not None
                    spike_times = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                    sampled_spike_times = spike_times[inds]
                    num_samples = recording.get_num_samples(segment_index=segment_index)
                    mask = (sampled_spike_times >= nbefore) & (sampled_spike_times < (num_samples - nafter))
                    inds = inds[mask]

                sel_spikes.append(inds)
            selected_spikes[unit_id] = sel_spikes
        return selected_spikes

    # endregion

    # region Quality metrics
    def compute_quality_metrics(self, metric_names, **params):
        """
        Compute quality metrics on waveform extractor.

        Parameters
        ----------
        self: WaveformExtractor
            Compute quality metrics on self
        metric_names: list or None
            List of quality metrics to compute.
        params: keyword arguments for quality metrics

        Returns
        -------
        metrics: pandas.DataFrame
            Data frame with the computed metrics
        """

        return self.compute_metrics(metric_names, **params)

    def compute_metrics(self, metric_names, peak_sign='neg',
                        max_spikes_for_nn=2000, n_neighbors=6, seed=None):
        """
        Computes quality metrics

        Parameters
        ----------
        metric_names: list or None
            List of quality metrics to compute.
        **kwargs: keyword arguments for quality metrics (TODO)
            max_spikes_for_nn: int
                maximum number of spikes to use per cluster in PCA metrics
            n_neighbors: int
                number of nearest neighbors to check membership of in PCA metrics
            seed: int
                seed for pseudorandom number generator used in PCA metrics (e.g. nn_isolation)

        Returns
        -------
        metrics: pd.DataFrame

        """

        params = dict(metric_names=[str(name) for name in metric_names],
                      peak_sign=peak_sign,
                      max_spikes_for_nn=int(max_spikes_for_nn),
                      n_neighbors=int(n_neighbors),
                      seed=int(seed) if seed is not None else None)

        metric_names = params["metric_names"]

        unit_ids = self.sorting.unit_ids
        metrics = pd.DataFrame(index=unit_ids)

        # Simple metrics not based on PCs
        _possible_pc_metric_names = ['isolation_distance', 'l_ratio', 'd_prime',
                                     'nearest_neighbor', 'nn_isolation', 'nn_noise_overlap']
        for name in metric_names:
            if name in _possible_pc_metric_names:
                print(f"Quality metric: {name} depends on PCA. Skipping metric")

            _metric_name_to_func = {
                # "num_spikes": compute_num_spikes,
                "firing_rate": CurationMetrics.compute_firing_rate,
                # "presence_ratio": compute_presence_ratio,
                "snr": CurationMetrics.compute_snrs,
                "isi_violation": CurationMetrics.compute_isi_violations,
                # "amplitude_cutoff": compute_amplitudes_cutoff,
            }
            func = _metric_name_to_func[name]

            # TODO add for params from different functions
            kwargs = {k: params[k] for k in ('peak_sign',)}

            res = func(self, **kwargs)
            if isinstance(res, dict):
                # res is a dict convert to series
                metrics[name] = pd.Series(res)
            else:
                # res is a namedtuple with several dict
                # so several columns
                for i, col in enumerate(res._fields):
                    metrics[col] = pd.Series(res[i])

        # region REMOVED
        # Original Purpose: Some metrics requires PCA
        # REASON FOR REMOVAL: The three metrics we are using (isi, firing rate, and signal-to-noise ratio) do not depend on PCA
        # # metrics based on PCs
        # pc_metric_names = [k for k in metric_names if k in _possible_pc_metric_names]
        # if len(pc_metric_names):
        #     if self.principal_component is None:
        #         raise ValueError('waveform_principal_component must be provied')
        #     kwargs = {k: self._params[k] for k in ('max_spikes_for_nn', 'n_neighbors', 'seed')}
        #     pc_metrics = calculate_pc_metrics(self.principal_component, metric_names=pc_metric_names, **kwargs)
        #     for col, values in pc_metrics.items():
        #         metrics[col] = pd.Series(values)
        # end region

        # Save to folder
        metrics.to_csv(self.folder / 'metrics.csv')

        return metrics

    # endregion

    # region Get waveforms and templates
    def get_template(self, unit_id, mode='average', sparsity=None):
        """
        Return template (average waveform).

        Parameters
        ----------
        unit_id: int or str
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

        if mode in self._template_cache:
            # already in the global cache
            templates = self._template_cache[mode]
            unit_ind = self.sorting.id_to_index(unit_id)
            template = templates[unit_ind, :, :]
            if sparsity is not None:
                chan_inds = self.recording.ids_to_indices(sparsity[unit_id])
                template = template[:, chan_inds]
            return template

        # compute from waveforms
        wfs = self.get_waveforms(unit_id, sparsity=sparsity)
        if mode == 'median':
            template = np.median(wfs, axis=0)
        elif mode == 'average':
            template = np.average(wfs, axis=0)
        elif mode == 'std':
            template = np.std(wfs, axis=0)
        return template

    def get_waveforms(self, unit_id, with_index=False, cache=True, memmap=True, sparsity=None):
        """
        Return waveforms for the specified unit id.

        Parameters
        ----------
        unit_id: int or str
            Unit id to retrieve waveforms for
        with_index: bool
            If True, spike indices of extracted waveforms are returned (default False)
        cache: bool
            If True, waveforms are cached to the self._waveforms dictionary (default False)
        memmap: bool
            If True, waveforms are loaded as memmap objects.
            If False, waveforms are loaded as np.array objects (default True)
        sparsity: dict or None
            If given, dictionary with unit ids as keys and channel sparsity by channel ids as values.
            The sparsity can be computed with the toolkit.get_template_channel_sparsity() function
            (make sure to use the default output='id' when computing the sparsity)

        Returns
        -------
        wfs: np.array
            The returned waveform (num_spikes, num_samples, num_channels)
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

        if sparsity is not None:
            assert unit_id in sparsity, f"Sparsity for unit {unit_id} is not in the sparsity dictionary!"
            chan_inds = self.recording.ids_to_indices(sparsity[unit_id])
            wfs = wfs[:, :, chan_inds]

        if with_index:  # with_index should be always False
            sampled_index = self.get_sampled_indices(unit_id)
            return wfs, sampled_index
        else:
            return wfs

    def precompute_templates(self, modes=('average', 'std')):
        """
        Precompute all template for different "modes":
          * average
          * std
          * median

        The results is cache in memory as 3d ndarray (nunits, nsamples, nchans)
        and also saved as npy file in the folder to avoid recomputation each time.
        """
        # TODO : run this in parralel

        dtype = self._params['dtype']
        unit_ids = self.sorting.unit_ids
        num_chans = self.recording.get_num_channels()

        for mode in modes:
            templates = np.zeros((len(unit_ids), self.nsamples, num_chans), dtype=dtype)
            self._template_cache[mode] = templates

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

                self._template_cache[mode][i, :, :] = arr

        for mode in modes:
            templates = self._template_cache[mode]
            template_file = self.folder / f'templates_{mode}.npy'
            np.save(template_file, templates)

    def select_units(self, unit_ids, new_folder):
        """
        Filters units by creating a new waveform extractor object in a new folder.
        (Create new folder with only selected waveform units)

        Parameters
        ----------
        unit_ids : list or array
            The unit ids to keep in the new WaveformExtractor object
        new_folder : Path
            The new folder where selected waveforms are copied

        Return
        ------
        we :  WaveformExtractor
            The newly create waveform extractor with the selected units
        """
        new_folder = Path(new_folder)
        if recompute_curation and new_folder.is_dir():
            shutil.rmtree(new_folder)
        # assert not new_folder.is_dir(), f"{new_folder} already exists!"
        new_folder.mkdir(parents=True)

        # Save unit_ids
        np.save(new_folder / "unit_ids.npy", np.array(unit_ids))

        # region REMOVED
        # ORIGINAL PURPOSE: Update recording and sorting with only curated units
        # REASON FOR REMOVAL: Unnecessary to save this data.
        # sorting = self.sorting.select_units(unit_ids)
        # # create new waveform extractor folder
        # shutil.copyfile(self.folder / "params.json",
        #                 new_folder / "params.json")
        # shutil.copyfile(self.folder / "recording.json",
        #                 new_folder / "recording.json")
        # sorting.dump(new_folder / 'sorting.json', relative_to=None)
        # endregion

        # create and populate waveforms folder
        new_waveforms_folder = new_folder / "waveforms"
        new_waveforms_folder.mkdir()

        waveforms_files = [f for f in (self.folder / "waveforms").iterdir() if f.suffix == ".npy"]
        for unit in unit_ids:
            for wf_file in waveforms_files:
                if f"waveforms_{unit}.npy" in wf_file.name or f'sampled_index_{unit}.npy' in wf_file.name:
                    shutil.copyfile(wf_file, new_waveforms_folder / wf_file.name)

        # region REMOVED
        # ORIGINAL PURPOSE: Updating extensions to extractor
        # REASON FOR REMOVAL: Extension paradigm is not used
        # for ext_name in self.get_available_extension_names():
        #     ext = self.load_extension(ext_name)
        #     ext.select_units(unit_ids, new_folder)
        # endregion

        we = WaveformExtractor.load_from_folder(self.recording, self.sorting, new_folder)
        # Needed for functions used to convert to MATLAB
        we._params = self._params
        # we.sorting.unit_ids = unit_ids

        return we

    def load_unit_ids(self):
        unit_ids_folder = self.folder / "unit_ids.npy"

        assert unit_ids_folder.exists(), "Cannot load curated waveform unit_ids since they have not been computed and saved yet"
        if unit_ids_folder.exists():
            return np.load(unit_ids_folder)
        # else:  # Loading all unit_ids
        #     test = [p for p in (self.folder/"waveforms").iterdir()]
        #     print(len(test))
        #     exit()

    # endregion


class CurationMetrics:
    @staticmethod
    def curate_firing_rates(recording, sorting):
        """
        Curate units based on firing rate. Units with firing rate below FR_MIN are not curated.

        Parameters
        ----------
        recording : se.MaxwellRecordingExtractor
            The recording object
        sorting : KilosortSortingExtractor
            The sorting object that contains unit_ids

        Returns
        -------
        curated_ids : np.array
            Array containing the curated units that are above firing rate threshold (FR_MIN)
        """
        fs = recording.get_sampling_frequency()

        # region REMOVED
        # ORIGINAL PURPOSE: Handle multi-segment recordings
        # REASON FOR REMOVAL: Recordings will always have 1 segment
        # seg_durations = [recording.get_num_samples(i) / fs for i in range(num_segs)]
        # total_duraion = np.sum(seg_durations)
        # endregion

        total_duration = recording.get_num_samples() / fs

        # firing_rates = {}
        curated_ids = []
        for unit_id in sorting.unit_ids:
            n_spikes = sorting.get_unit_spike_train(unit_id=unit_id).size
            firing_rate = n_spikes / total_duration
            if firing_rate > fr_thresh:
                curated_ids.append(unit_id)
            # firing_rates[unit_id] = n_spikes / total_duration

        # return firing_rates
        return np.asarray(curated_ids)

    @staticmethod
    def curate_isi_violations(recording, sorting, isi_threshold_ms=1.5, min_isi_ms=0, **kwargs):
        """
        Curate units based on ISI violations. Units with violations greater than ISI_VIOL_MAX are
        removed from curation

        It computes several metrics related to isi violations:
            * violation_rate_ratio: the relative firing rate of the hypothetical neurons that are generating the ISI
                                    violations. Described in [1]. See Notes.
            * violation_rate: number of ISI violations divided by total rate
            * violations_num: number of ISI violations

        [1] Hill et al. (2011) J Neurosci 31: 8699-8705

        Parameters
        ----------
        recording : se.MaxwellRecordingExtractor
            The recording object
        sorting : KilosortSortingExtractor
            The sorting object that contains unit_ids
        isi_threshold_ms : float
            Threshold for classifying adjacent spikes as an ISI violation. This is the biophysical refractory period
            (default=1.5)
        min_isi_ms : float
            Minimum possible inter-spike interval (default=0). This is the artificial refractory period enforced
            by the data acquisition system or post-processing algorithms

        Returns
        -------
        curated_its: np.array
            Array containing the curated units that do not violate ISI

        Notes
        -----
        You can interpret an ISI violations ratio value of 0.5 as meaning that contaminated spikes are occurring at roughly
        half the rate of "true" spikes for that unit. In cases of highly contaminated units, the ISI violations value can
        sometimes be even greater than 1.

        Originally written in Matlab by Nick Steinmetz (https://github.com/cortex-lab/sortingQuality) and
        converted to Python by Daniel Denman.
        """

        # region REMOVED
        # ORIGINAL PURPOSE: Determine total_duration when there are multiple segments
        # REASON FOR REMOVAL: Recording will always have 1 segment
        # num_segs = sorting.get_num_segments()
        # fs = recording.get_sampling_frequency()
        #
        # seg_durations = [recording.get_num_samples(i) / fs for i in range(num_segs)]
        # total_duration = np.sum(seg_durations)
        # region REMOVED

        fs = recording.get_sampling_frequency()
        total_duration = recording.get_num_samples() / fs

        isi_threshold_s = isi_threshold_ms / 1000
        min_isi_s = min_isi_ms / 1000
        isi_threshold_samples = int(isi_threshold_s * fs)

        # isi_violations_rate = {}
        # isi_violations_count = {}
        # isi_violations_ratio = {}

        curated_ids = []
        # All units converted to seconds
        for unit_id in sorting.unit_ids:
            # region REMOVED
            # ORIGINAL PURPOSE: Handle multisegment recordings
            # REASON FOR REMOVAL: Recording will have 1 segment always
            # num_violations = 0
            # num_spikes = 0
            # for segment_index in range(num_segs):
            #     spike_train = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            #     isis = np.diff(spike_train)
            #     num_spikes += len(spike_train)
            #     num_violations += np.sum(isis < isi_threshold_samples)
            # endregion

            spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
            num_spikes = len(spike_train)

            isis = np.diff(spike_train)

            violation_num = np.sum(isis < isi_threshold_samples)

            violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s)
            total_rate = num_spikes / total_duration
            violation_rate = violation_num / violation_time
            violation_rate_ratio = violation_rate / total_rate
            # isi_violations_ratio[unit_id] = violation_rate / total_rate
            # isi_violations_rate[unit_id] = num_violations / total_duration
            # isi_violations_count[unit_id] = num_violations

            if violation_rate_ratio < isi_viol_thresh:
                curated_ids.append(unit_id)

        # res = namedtuple('isi_violaion', ['isi_violations_ratio', 'isi_violations_rate', 'isi_violations_count'])
        # return res(isi_violations_ratio, isi_violations_rate, isi_violations_count)
        return np.asarray(curated_ids)

    @staticmethod
    def curate_snrs(recording, sorting, peak_sign='neg', **kwargs):
        """
        Curate units based on signal-to-noise ratio

        Parameters
        ----------
        recording : se.MaxwellRecordingExtractor
            The recording object
        sorting : KilosortSortingExtractor
            The sorting object that contains unit_ids
        peak_sign: 'neg' or 'pos'
            Which extremum to be considered the peak
        kwargs

        Returns
        -------
        curated_ids: np.array
            Array containing the curated units that do not violate signal-to-noise ratio
        """
        extremum_channels = CurationMetrics.get_template_extremum_channel_and_amplitudes_from_sorting(recording, sorting, peak_sign=peak_sign)

        return_scaled_corrected = return_scaled
        if return_scaled_corrected:
            # Check if recording has scaled values:
            if not recording.has_scaled_traces():
                print("Setting 'RETURN_SCALED' to False")
                return_scaled_corrected = False

        print("Extracting noise levels from recording")
        stopwatch = Stopwatch()
        noise_levels = CurationMetrics.get_noise_levels(recording, return_scaled=False, **kwargs)
        stopwatch.log_time("Done extracting noise.")

        # make a dict to access by chan_id
        # noise_levels = dict(zip(channel_ids, noise_levels))
        # snrs = {}
        curated_ids = []
        # channel_gains = recording_raw.get_channel_gains()
        for unit_id in sorting.unit_ids:
            chan_ind, amplitude = extremum_channels[unit_id]
            noise = noise_levels[chan_ind]
            # amplitude *= channel_gains[chan_ind]
            snr = np.abs(amplitude) / noise
            # print(f"{amplitude=:.3f} {noise=:.3f} {snr=:.3f}")
            if snr > snr_thresh:
                curated_ids.append(unit_id)
            # snrs[unit_id] = np.abs(amplitude) / noise

        return np.asarray(curated_ids)

    # region Functions needed for compute_snrs and converting to matlab
    @staticmethod
    def get_template_extremum_channel_from_waveforms(waveform_extractor, peak_sign='neg', outputs='id'):
        """
        Compute the channel with the extremum peak for each unit.

        Parameters
        ----------
        waveform_extractor: WaveformExtractor
            The waveform extractor
        peak_sign: str
            Sign of the template to compute best channels ('neg', 'pos', 'both')
        outputs: str
            * 'id': channel id
            * 'index': channel index

        Returns
        -------
        extremum_channels: dict
            Dictionary with unit ids as keys and extremum channels (id or index based on 'outputs')
            as values
        """

        # region Functions needed
        def get_template_amplitudes(waveform_extractor, peak_sign='neg', mode='extremum'):
            """
            Get amplitude per channel for each unit.

            Parameters
            ----------
            waveform_extractor: WaveformExtractor
                The waveform extractor
            peak_sign: str
                Sign of the template to compute best channels ('neg', 'pos', 'both')
            mode: str
                'extremum':  max or min
                'at_index': take value at spike index

            Returns
            -------
            peak_values: dict
                Dictionary with unit ids as keys and template amplitudes as values
            """
            assert peak_sign in ('both', 'neg', 'pos')
            assert mode in ('extremum', 'at_index')
            unit_ids = waveform_extractor.sorting.unit_ids

            before = waveform_extractor.nbefore

            peak_values = {}

            for unit_id in unit_ids:
                template = waveform_extractor.get_template(unit_id, mode='average')

                if mode == 'extremum':
                    if peak_sign == 'both':
                        values = np.max(np.abs(template), axis=0)
                    elif peak_sign == 'neg':
                        values = -np.min(template, axis=0)
                    elif peak_sign == 'pos':
                        values = np.max(template, axis=0)
                elif mode == 'at_index':
                    if peak_sign == 'both':
                        values = np.abs(template[before, :])
                    elif peak_sign == 'neg':
                        values = -template[before, :]
                    elif peak_sign == 'pos':
                        values = template[before, :]

                peak_values[unit_id] = values

            return peak_values
        # endregion

        unit_ids = waveform_extractor.sorting.unit_ids
        channel_ids = waveform_extractor.recording.channel_ids

        peak_values = get_template_amplitudes(waveform_extractor, peak_sign=peak_sign)
        extremum_channels_id = {}
        extremum_channels_index = {}
        for unit_id in unit_ids:
            max_ind = np.argmax(peak_values[unit_id])
            extremum_channels_id[unit_id] = channel_ids[max_ind]
            extremum_channels_index[unit_id] = max_ind

        if outputs == 'id':
            return extremum_channels_id
        elif outputs == 'index':
            return extremum_channels_index

    @staticmethod
    def get_template_extremum_channel_and_amplitudes_from_sorting(recording, sorting, peak_sign='neg'):
        """
        Compute the channel with the extremum peak for each unit and the amplitude.

        Parameters
        ----------
        recording: se.MaxwellRecordingExtractor
            The recording extractor
        sorting: KilosortSortingExtractor
            The sorting extractor
        peak_sign: str
            Sign of the template to compute best channels ('neg', 'pos', 'both')
        outputs: str
            * 'id': channel id
            * 'index': channel index

        Returns
        -------
        extremum_channels: dict
            Dictionary with unit ids as keys and tuple (extremem_channel_id, unit_amplitude)
            as values
        """

        assert peak_sign in ('both', 'neg', 'pos')

        templates = sorting.get_templates_raw()
        # amplitudes are np.array with shape (n_units, n_channels)
        if peak_sign == 'both':
            amplitudes = np.max(np.abs(templates), axis=1)
        elif peak_sign == 'neg':
            amplitudes = -np.min(templates, axis=1)
        elif peak_sign == 'pos':
            amplitudes = np.max(templates, axis=1)

        # channel_ids = recording.channel_ids

        extremum_channels = {}
        for ind, unit_id in enumerate(sorting.unit_ids):
            kilosort_max_ind = np.argmax(amplitudes[ind])
            max_ind = sorting.channel_map[kilosort_max_ind]
            max_amp = np.max(amplitudes[ind])

            # extremum_channels[unit_id] = (channel_ids[max_ind], max_amp)
            extremum_channels[unit_id] = (max_ind, max_amp)

        return extremum_channels

    @staticmethod
    def get_template_extremum_amplitude(waveform_extractor, peak_sign='neg'):
        """
        Computes amplitudes on the best channel.

        Parameters
        ----------
        waveform_extractor: WaveformExtractor
            The waveform extractor
        peak_sign: str
            Sign of the template to compute best channels ('neg', 'pos', 'both')

        Returns
        -------
        amplitudes: dict
            Dictionary with unit ids as keys and amplitudes as values
        """
        unit_ids = waveform_extractor.sorting.unit_ids

        before = waveform_extractor.nbefore

        extremum_channels_ids = CurationMetrics.get_template_extremum_channel_from_waveforms(waveform_extractor, peak_sign=peak_sign)

        unit_amplitudes = {}
        for unit_id in unit_ids:
            template = waveform_extractor.get_template(unit_id, mode='average')
            chan_id = extremum_channels_ids[unit_id]
            chan_ind = waveform_extractor.recording.id_to_index(chan_id)
            unit_amplitudes[unit_id] = template[before, chan_ind]

        return unit_amplitudes

    @staticmethod
    def get_noise_levels(recording, return_scaled=True, **random_chunk_kwargs):
        """
        Estimate noise for each channel using MAD methods.

        Internally it sample some chunk across segment.
        And then, it use MAD estimator (more robust than STD)

        """

        # region Functions needed
        def get_random_data_chunks(recording, return_scaled=False, num_chunks_per_segment=20, chunk_size=10000, seed=0):
            """
            Exctract random chunks across segments

            This is used for instance in get_noise_levels() to estimate noise on traces.

            Parameters
            ----------
            recording: BaseRecording
                The recording to get random chunks from
            return_scaled: bool
                If True, returned chunks are scaled to uV
            num_chunks_per_segment: int
                Number of chunks per segment
            chunk_size: int
                Size of a chunk in number of frames
            seed: int
                Random seed

            Returns
            -------
            chunk_list: np.array
                Array of concatenate chunks per segment
            """
            # TODO: if segment have differents length make another sampling that dependant on the lneght of the segment
            # Should be done by chnaging kwargs with total_num_chunks=XXX and total_duration=YYYY
            # And randomize the number of chunk per segment wieighted by segment duration

            chunk_list = []
            for segment_index in range(recording.get_num_segments()):
                length = recording.get_num_frames(segment_index)
                random_starts = np.random.RandomState(seed=seed).randint(0,
                                                                         length - chunk_size, size=num_chunks_per_segment)
                for start_frame in random_starts:
                    chunk = recording.get_traces(start_frame=start_frame,
                                                 end_frame=start_frame + chunk_size,
                                                 segment_index=segment_index,
                                                 return_scaled=return_scaled)
                    chunk_list.append(chunk)
            return np.concatenate(chunk_list, axis=0)

        # endregion
        random_chunks = get_random_data_chunks(recording, return_scaled=return_scaled, **random_chunk_kwargs)
        med = np.median(random_chunks, axis=0, keepdims=True)
        noise_levels = np.median(np.abs(random_chunks - med), axis=0) / 0.6745
        return noise_levels


# endregion


# region Recording and Sorting Segments
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


# region SpikeInterface Utilities
class ShellScript:
    """
    Code for running shell scripts
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
            * "CHUNK_MEMORY": total memory per chunk per worker
            * "TOTAL_MEMORY": total memory over all workers.

        If chunk_size/CHUNK_MEMORY/TOTAL_MEMORY are all None then there is no chunk computing
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
                raise ValueError('For N_JOBS >1 you must specify TOTAL_MEMORY or chunk_size or CHUNK_MEMORY')

        return chunk_size

    @staticmethod
    def _mem_to_int(mem):
        # Converts specified memory (e.g. 4G) to integer number
        _exponents = {'k': 1e3, 'M': 1e6, 'G': 1e9}

        suffix = mem[-1]
        assert suffix in _exponents
        mem = int(float(mem[:-1]) * _exponents[suffix])
        return mem


class ChunkRecordingExecutor:
    """
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
        Size of each chunk in number of samples. If 'TOTAL_MEMORY' or 'CHUNK_MEMORY' are used, it is ignored.
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
            print(self.job_name, 'with', 'N_JOBS', self.n_jobs, ' chunk_size', self.chunk_size)

    def run(self):
        """
        Runs the defined jobs.
        """
        all_chunks = JobUtils.divide_recording_into_chunks(self.recording, self.chunk_size)

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
                                     initializer=worker_initializer,
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


class WorkerChunk:
    """
    Functions used by WaveformExtractor + ChunkRecordingExecutor
    """

    @staticmethod
    def waveform_extractor_chunk(segment_index, start_frame, end_frame, worker_ctx):
        # recover variables of the worker
        recording = worker_ctx['recording']
        sorting = worker_ctx['sorting']
        wfs_memmap_files = worker_ctx['wfs_memmap_files']
        selected_spikes = worker_ctx['selected_spikes']
        selected_spike_times = worker_ctx['selected_spike_times']
        nbefore = worker_ctx['nbefore']
        nafter = worker_ctx['nafter']
        return_scaled = worker_ctx['RETURN_SCALED']
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
            start = min(st[0] for _, _, st in to_extract.values()) - nbefore
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
                    wfs[pos, :, :] = traces[st - start - nbefore:st - start + nafter, :]

    @staticmethod
    def init_worker_waveform_extractor(recording, sorting, wfs_memmap,
                                       selected_spikes, selected_spike_times, nbefore, nafter, return_scaled):
        # create a local dict per worker
        worker_ctx = {}
        # region REMOVED
        # ORIGINAL PURPOSE: Load recording object from dictionary
        # REASON FOR REMOVAL: The object itself, not a dictionary, will be passed
        # if isinstance(recording, dict):
        #     from spikeinterface.core import load_extractor
        #     recording = load_extractor(recording)
        # endregion
        worker_ctx['recording'] = recording

        # region REMOVED
        # ORIGINAL PURPOSE: Load recording object from dictionary
        # REASON FOR REMOVAL: The object itself, not a dictionary, will be passed
        # if isinstance(sorting, dict):
        #     from spikeinterface.core import load_extractor
        #     sorting = load_extractor(sorting)
        # endregion
        worker_ctx['sorting'] = sorting

        worker_ctx['wfs_memmap_files'] = wfs_memmap
        worker_ctx['selected_spikes'] = selected_spikes
        worker_ctx['selected_spike_times'] = selected_spike_times
        worker_ctx['nbefore'] = nbefore
        worker_ctx['nafter'] = nafter
        worker_ctx['RETURN_SCALED'] = return_scaled

        num_seg = sorting.get_num_segments()
        unit_cum_sum = {}
        for unit_id in sorting.unit_ids:
            # spike per segment
            n_per_segment = [selected_spikes[unit_id][i].size for i in range(num_seg)]
            cum_sum = [0] + np.cumsum(n_per_segment).tolist()
            unit_cum_sum[unit_id] = cum_sum
        worker_ctx['unit_cum_sum'] = unit_cum_sum

        return worker_ctx


global _worker_ctx
global _func


def worker_initializer(func, init_func, init_args):
    global _worker_ctx
    _worker_ctx = init_func(*init_args)
    global _func
    _func = func


class JobUtils:
    @staticmethod
    def divide_recording_into_chunks(recording, chunk_size):
        all_chunks = []
        for segment_index in range(recording.get_num_segments()):
            num_frames = recording.get_num_samples(segment_index)
            chunks = JobUtils.divide_segment_into_chunks(num_frames, chunk_size)
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


# endregion


# region ProcessPoolExecutor
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
                            'got %d' % (len(args) - 1))

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


# region spikesort_matlab.py
class Stopwatch:
    def __init__(self):
        self._time_start = time.time()

    def log_time(self, text=None):
        if text is None:
            print(f"Time: {time.time() - self._time_start:.2f}s")
        else:
            print(f"{text} Time: {time.time() - self._time_start:.2f}s")


def print_stage(text):
    text = str(text)

    num_chars = 50
    char = "="
    indent = int((num_chars - len(text)) / 2)

    print()
    print(num_chars * char)
    print(indent * " " + text)
    print(num_chars * char)


def load_recording(rec_path):
    print_stage("LOADING MAXWELL RECORDING")
    print(f"Recording path: {rec_path}")
    try:
        stopwatch = Stopwatch()
        if str(rec_path).endswith(".raw.h5"):
            rec = se.MaxwellRecordingExtractor(rec_path)
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
        rec = se.MaxwellRecordingExtractor(rec_file)
        recordings.append(rec)
        print(f"{rec_name}: DURATION: {rec.get_num_frames() / rec.get_sampling_frequency()} s -- "
              f"NUM. CHANNELS: {rec.get_num_channels()}")
    rec = si_segmentutils.concatenate_recordings(recordings)
    print(f"Done concatenating {len(recordings)} recordings")
    return rec


def get_paths(rec_path, inter_path):
    inter_path = Path(inter_path)

    rec_path = Path(rec_path)
    print_stage("PROCESSING RECORDING")
    print(f"Recording path: {rec_path}")

    if recompute_sorting and inter_path.exists():
        for path in inter_path.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Deleted {path}")
            elif rewrite_recording and path.name == "recording.dat":
                path.unlink()
                print(f"Deleted {path}")

    sorter = "kilosort2"

    output_folder = inter_path / 'sorted' / sorter

    cache_folder = inter_path / 'cache'
    cache_folder.mkdir(parents=True, exist_ok=True)

    # region REMOVED
    # ORIGINAL PURPOSE: Path for storing matplotlib.pyplot figures
    # REASON FOR REMOVAL: Unnecessary
    # figures_folder = inter_path / 'figures'
    # figures_folder.mkdir(parents=True, exist_ok=True)

    # Needed for Kilosort2?

    tmp_folder = cache_folder / 'tmp' / sorter
    tmp_folder.mkdir(parents=True, exist_ok=True)

    return rec_path, inter_path, output_folder, cache_folder


def write_recording(recording, inter_path, verbose=False):
    print_stage("WRITING BINARY RECORDING")
    stopwatch = Stopwatch()

    recording_dat_path = inter_path / 'recording.dat'
    print(recording_dat_path)
    print("Using bandpass filter")

    rec_filtered = st.preprocessing.bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max)

    # Write binary file (recording.dat) which is the recording format that Kilosort expects
    # region REMOVED
    # ORIGINAL PURPOSE: Store recording data as .json
    # REASON FOR REMOVAL: Unnecessary
    # rec_file = output_folder / 'spikeinterface_recording.json'
    # if recording.is_dumpable:
    #     recording.dump_to_json(rec_file)
    # else:
    #     d = {'warning': 'The recording is not dumpable'}
    #     rec_file.write_text(json.dumps(d, indent=4), encoding='utf8')
    # endregion

    if not recording_dat_path.exists():
        BinaryRecordingExtractor.write_recording(rec_filtered, file_paths=recording_dat_path,
                                                 dtype='int16', total_memory=total_memory,
                                                 n_jobs=n_jobs, verbose=False, progress_bar=verbose)
    else:
        print(f"Skipping writing recording.dat\nUsing existing {recording_dat_path} as recording file")
    stopwatch.log_time("Done writing recording.")
    return rec_filtered, recording_dat_path


def spike_sort(rec_cache, rec_path, recording_dat_path, output_folder):
    print_stage("SORTING")
    stopwatch = Stopwatch()
    sorting = -1

    kilosort = RunKilosort()
    try:
        if output_folder.is_dir():
            print('Skipping', rec_path, ' since already sorted')
            sorting = kilosort.get_result_from_folder(output_folder)
            # sorting = si.load_extractor(sorting_raw_folder)
        else:
            output_folder.mkdir(parents=True)
            time_sorting_start = time.time()
            sorting = kilosort.run(recording=rec_cache, recording_dat_path=recording_dat_path, output_folder=output_folder, verbose=True)
            print(f"\n\nSpike sorting elapsed time {time.time() - time_sorting_start} s")

    except Exception as e:
        print(f"Kilosort2 failed on recording {rec_path} because of {e}")

    stopwatch.log_time("Done sorting.")

    return sorting


def extract_waveforms(recording, sorting, folder, load_if_exists=False, precompute_template=('average',), dtype=None):
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
    load_if_exists: bool
        If True and waveforms have already been extracted in the specified folder, they are loaded
        and not recomputed.
        If False and waveforms have already been extracted in the specified folder, they are deleted
        and recomputed.
    precompute_template: None or list
        Precompute average/std/median for template. If None not precompute.
    WAVEFORMS_MS_BEFORE: float
        Time in ms to cut before spike peak
    WAVEFORMS_MS_AFTER: float
        Time in ms to cut after spike peak
    max_spikes_per_unit: int or None
        Number of spikes per unit to extract waveforms from (default 500).
        Use None to extract waveforms for all spikes
    RETURN_SCALED: bool
        If True and recording has gain_to_uV/offset_to_uV properties, waveforms are converted to uV.
    dtype: dtype or None
        Dtype of the output waveforms. If None, the recording dtype is maintained.
    use_relative_path: bool
        If True, the recording and sorting paths are relative to the waveforms folder.
        This allows portability of the waveform folder provided that the relative paths are the same,
        but forces all the data files to be in the same drive.
        Default is False.

    {}

    Returns
    -------
    we: WaveformExtractor
        The WaveformExtractor object

    """
    precompute_template = None
    print_stage("EXTRACTING WAVEFORMS")
    stopwatch = Stopwatch()

    folder_waveforms = folder / "waveforms"
    if not load_if_exists and folder_waveforms.is_dir():
        shutil.rmtree(folder_waveforms)
        print(f"Deleted {folder}")

    if load_if_exists and folder.is_dir():  # Load saved waveform extractor
        print("Loading waveforms from folder")
        we = WaveformExtractor.load_from_folder(recording, sorting, folder)
        we.set_params(dtype=dtype)
    else:  # Create new waveform extractor
        print("Creating new waveform extractor")
        we = WaveformExtractor.create(recording, sorting, folder)
        we.set_params(dtype=dtype)
        we.run_extract_waveforms(n_jobs=n_jobs, total_memory=total_memory, progress_bar=True)
        # np.save(folder / "unit_ids.npy", sorting.unit_ids)
        if precompute_template is not None:
            we.precompute_templates(modes=precompute_template)
    stopwatch.log_time("Done extracting waveforms.")
    return we


def curate_units(recording, sorting, cache_folder):

    print_stage("CURATING UNITS")
    total_stopwatch = Stopwatch()
    curated_path = cache_folder / "waveforms_curated"
    curated_units_path = curated_path / "unit_ids.npy"
    if curated_units_path.exists() and not recompute_curation:
        print("Skipping unit curation since already curated.")
        print("Loading saved curated units.")
        stopwatch = Stopwatch()
        sorting.unit_ids = np.load(str(curated_units_path))
        stopwatch.log_time("Done loading.")
        return

    if not auto_curate:
        print("\nAUTO_CURATE is set to False, so skipping data curation.")
        return

    print(f'N units before curation: {len(sorting.unit_ids)}')

    print_stage("FIRING RATE")
    stopwatch = Stopwatch()
    sorting.unit_ids = CurationMetrics.curate_firing_rates(recording, sorting)
    print(f'N units after firing rate curation: {len(sorting.unit_ids)}')
    stopwatch.log_time()

    print_stage("ISI VIOLATION RATE")
    stopwatch = Stopwatch()
    sorting.unit_ids = CurationMetrics.curate_isi_violations(recording, sorting)
    print(f'N units after firing rate curation: {len(sorting.unit_ids)}')
    stopwatch.log_time()

    print_stage("SIGNAL-TO-NOISE RATIO")
    stopwatch = Stopwatch()
    sorting.unit_ids = CurationMetrics.curate_snrs(recording, sorting)
    print(f'N units after signal-to-noise ratio curation: {len(sorting.unit_ids)}')
    stopwatch.log_time()

    print_stage("SAVING CURATED UNIT IDS")
    print(f"Saving to path {curated_units_path}")
    stopwatch = Stopwatch()
    curated_path.mkdir(parents=True)
    np.save(curated_units_path, sorting.unit_ids)
    stopwatch.log_time("Done saving.")

    total_stopwatch.log_time("Done curating units. Total")

    # we_raw = extract_waveforms(recording, sorting, cache_folder / "waveforms_raw",
    #                            load_if_exists=True)

    # print_stage("COMPUTING CURATION METRICS")
    # stopwatch = Stopwatch()
    # waveforms_curated = we_raw.compute_quality_metrics(metric_names=["firing_rate", "isi_violation", "snr"])
    #
    # print("Done computing metrics." + stopwatch.log_time())
    # print_stage("SELECTING UNITS")
    # stopwatch = Stopwatch()
    # print(f'N units before curation: {len(sorting.unit_ids)}')
    #
    # # Curation based on firing rate
    # if FR_MIN is not None:
    #     print(f"Curation based on firing rate")
    #     fr_query = f"firing_rate > {FR_MIN}"
    #     waveforms_curated = waveforms_curated.query(fr_query)
    #     print(f'N units after num spikes curation: {len(waveforms_curated)}\n')
    #
    # # Curation based on interspike-interval-violation
    # if ISI_VIOL_MAX is not None:
    #     if len(waveforms_curated) > 0:
    #         print(f"Curation based on isi violation ratio")
    #         isi_query = f"isi_violations_rate < {ISI_VIOL_MAX}"
    #         waveforms_curated = waveforms_curated.query(isi_query)
    #         print(f'N units after ISI curation: {len(waveforms_curated)}\n')
    #     else:
    #         print("No units remain after curation")
    #
    # # Curation based on signal-to-noise ratio
    # if SNR_MIN is not None:
    #     if len(waveforms_curated) > 0:
    #         print(f"Curation based on SNR")
    #         snr_query = f"snr > {SNR_MIN}"
    #         waveforms_curated = waveforms_curated.query(snr_query)
    #         print(f'N units after SNR curation: {len(waveforms_curated)}\n')
    #     else:
    #         print("No units remain after curation")
    #
    # # The indices of values to keep
    # keep_unit_ids = waveforms_curated.index.values
    # # Save only the curated waveforms to a new file
    # print(f"Saving curated waveforms to folder {curated_folder}")
    # we_curated = we_raw.select_units(unit_ids=keep_unit_ids, new_folder=curated_folder)
    # print("Done curating units." + stopwatch.log_time())
    # return we_curated


def process_recording(rec_path, inter_path):
    # Start time for entire process
    stopwatch = Stopwatch()

    # Get Paths
    rec_path, inter_path, output_folder, cache_folder = get_paths(rec_path, inter_path)

    # Load Recording
    recording_raw = load_recording(rec_path)
    if recording_raw == -1:  # Could not load recording
        return "Could not load recording. Moving on to next recording."

    # Filter and Dumping
    recording_filtered, recording_dat_path = write_recording(recording_raw, inter_path, verbose=True)

    # Spike Sorting
    sorting = spike_sort(rec_cache=recording_filtered, rec_path=rec_path, recording_dat_path=recording_dat_path, output_folder=output_folder)
    if sorting == -1:  # Could not sort recording
        return "Could not sort recording. Moving on to next recording"

    # Curating Data
    curate_units(recording_raw, sorting, cache_folder)

    # Extracting waveforms
    waveform_extractor = extract_waveforms(recording_filtered, sorting, cache_folder / "waveforms_curated",
                                           load_if_exists=not recompute_waveforms)

    print_stage("DONE PROCESSING RECORDING")
    print(rec_path)
    stopwatch.log_time("Total")

    return waveform_extractor


def convert_to_matlab(waveform_extractor, rec_path, matlab_path):
    print_stage("CONVERTING TO MATLAB")
    stopwatch = Stopwatch()

    # Get matlab file path
    if matlab_path.endswith(".mat"):
        mat_file = Path(matlab_path)
    else:
        name = str(rec_path).split(r"/")[-1].split(".")[0] + "_sorted.mat"
        mat_file = Path(matlab_path) / name

    if mat_file.exists():
        if override_matlab:
            mat_file.unlink()
            print(f"Deleted {mat_file}")
        else:
            print(f"Skipping {mat_file} since already created")
            return
    # Make parent file
    mat_file.parent.mkdir(exist_ok=True, parents=True)

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    unit_ids = sorting.unit_ids
    # unit_ids = waveform_extractor.load_unit_ids()
    # sorting.unit_ids = unit_ids
    mdict = {"units": [], "locations": recording.get_channel_locations(), "fs": recording.get_sampling_frequency()}

    # Get max channels
    # max_channels = CurationMetrics.get_template_extremum_channel_from_waveforms(waveform_extractor)
    extremum_channels = CurationMetrics.get_template_extremum_channel_and_amplitudes_from_sorting(recording, sorting)

    # Get channel locations
    locations = recording.get_channel_locations()

    # Get electrodes
    if save_electrodes:
        electrodes = recording.get_property('electrode')

    for u in unit_ids:

        unit_dict = {}

        # max_channel_idx = recording.id_to_index(max_channels[u])
        max_channel_idx = extremum_channels[u][0]

        spike_train = sorting.get_unit_spike_train(u)

        max_location = locations[max_channel_idx]

        template = waveform_extractor.get_template(u)

        unit_dict["spike_train"] = spike_train
        unit_dict["x_max"] = max_location[0]
        unit_dict["y_max"] = max_location[1]
        unit_dict["template"] = template
        unit_dict["unit_id"] = u

        if save_electrodes:
            electrode = electrodes[max_channel_idx]
            unit_dict["electrode"] = electrode

        mdict["units"].append(unit_dict)

    savemat(mat_file, mdict)
    print(f"Saved to {mat_file}.")
    stopwatch.log_time()


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
        stopwatch.log_time("TOTAL")
        print("\n\n")


if __name__ == "__main__":
    print_stage("STARTING SCRIPT")
    main()
# endregion
