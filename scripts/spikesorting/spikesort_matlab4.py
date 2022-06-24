######################################################
################  FOLDER LOCATIONS ###################
######################################################
# RECORDING_FILES, INTERMEDIATE_FOLDERS, and RESULTS_FOLDERS
# must all have the same length

# List of data files (.raw.h5) you want to spike sort
# If path is a folder with multiple recording files,
# they will be concatenated in natural ordering.
RECORDING_FILES = [
    # "/home/maxlim/SpikeSorting/data/DL/recordings/maxone_2950.raw.h5",
    "/home/maxlim/SpikeSorting/data/maxone/200123/2953/recordings"
    # "/home/maxlim/SpikeSorting/data/DL/recordings/maxone_2953.raw.h5",
    # "/home/maxlim/SpikeSorting/data/DL/recordings/maxone_2954.raw.h5",
    # "/home/maxlim/SpikeSorting/data/DL/recordings/maxone_2957.raw.h5",
    # "/home/maxlim/SpikeSorting/data/DL/recordings/maxone_5116.raw.h5",
    # "/home/maxlim/SpikeSorting/data/DL/recordings/maxone_5118.raw.h5"
]
# List of intermediate folders where Kilosort2 sorting results, waveforms, and graphs are saved
INTERMEDIATE_FOLDERS = [
    # "/home/maxlim/SpikeSorting/data/test/220608/maxone_2950",
    "/home/maxlim/SpikeSorting/data/dated/220623/maxone_2953_concat_all",
    # "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_2954",
    # "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_2957",
    # "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_5116",
    # "/home/maxlim/SpikeSorting/data/DL/sorted/processed/maxone_5118",
]

# List of output folders where final matlab files and figures are saved.
# Matlab files will have the same name as recording files but will end with _sorted.mat
# Figures will have the same name as recording files but will end with _DESCRIPTIVE_NAME.png (e.g. maxone_2953_raster_plot.png)
RESULTS_FOLDERS = [
    # "/home/maxlim/SpikeSorting/data/test/220608/maxone_2950",
    "/home/maxlim/SpikeSorting/data/dated/220623/maxone_2953_concat",
    # "/home/maxlim/SpikeSorting/data/test/220608/maxone_2954",
    # "/home/maxlim/SpikeSorting/data/test/220608/maxone_2957",
    # "/home/maxlim/SpikeSorting/data/test/220608/maxone_5116",
    # "/home/maxlim/SpikeSorting/data/test/220608/maxone_5118",
]

######################################################
###############  KILOSORT PARAMETERS  ################
######################################################
# Set Kilosort2's path
KILOSORT_PATH = "/home/maxlim/SpikeSorting/Kilosort2"  # "/home/vandermolen/Kilosort"
# Set plugin path
HDF5_PLUGIN_PATH = '/home/maxlim/SpikeSorting/extra_libraries/'  # '/home/vandermolen/SpikeSorting/extra_libraries/'

# If True, script will automatically run the following commands which setup MATLAB environment ofr pod-gpu.cnsi.ucsb.edu:
"""
export MW_NVCC_PATH=/usr/local/cuda-10.1/bin
module load MatLab/R2021b 
module load cuda/10.1
"""
AUTO_SETUP_POD_ENV = True

# Kilosort2 params
KILOSORT_PARAMS = {
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
# (5/19/2022) If False and exists and new recording is specified in RECORDING_FILES, it will not be computed and the old data will be used.
RECOMPUTE_RECORDING = True  # Refers to the .dat recording file created for Kilosort2. If True, the 3 other recompute variables become True too
RECOMPUTE_SORTING = False  # If True, entire intermediate_folder will be deleted
REEXTRACT_WAVEFORMS = False
RECURATE_SPIKES = True
RECURATE_WAVEFORMS = True
OVERRIDE_MATLAB = True
RECREATE_FIGURES = True

######################################################
##################  COPY SCRIPT?  ####################
######################################################
# If True, a copy of the script will be saved to each path in 'INTERMEDIATE_FOLDERS' with name prepended with "year_month_day_hourminute"
SAVE_SCRIPT = True

######################################################
##########  PARALLEL PROCESSING PARAMETERS  ##########
######################################################
# N_JOBS and TOTAL_MEMORY affects how bandpass filter is applied to raw recording, causing different .dat file
# Number of jobs to use for converting raw recording, extracting waveforms, and curation
N_JOBS = 64
# Total RAM to use for converting raw extracting waveforms and curation
TOTAL_MEMORY = "1G"

######################################################
############  BANDPASS FILTER PARAMETERS  ############
######################################################
FREQ_MIN = 300
FREQ_MAX = 6000

######################################################
###############  WAVEFORM PARAMETERS  ################
######################################################
# NOTE: If waveform parameters are change, set REEXTRACT_WAVEFORMS to True
#       Otherwise, old parameters will be used to analyze waveforms

# ms before and after waveform trough (negative peak) to extract
WAVEFORMS_MS_BEFORE = 5.
WAVEFORMS_MS_AFTER = 5.
# If the positive peak of the waveform or template is POS_PEAK_THRESH times as large as
# the absolute value of negative peak, the positive peak is used for centering the waveform, amplitude,
# and finding the max channel. Otherwise, the negative peak is used
POS_PEAK_THRESH = 2
# If True and recording has gain_to_uV/offset_to_uV properties, waveforms are converted to uV
RETURN_SCALED = True
# Maximum number of waveforms (spikes) per unit to compute templates and extract (None-> all spikes are used)
MAX_WAVEFORMS_PER_UNIT = None
# Data type of waveforms (None-> the recording's datatype is maintained)
WAVEFORM_DTYPE = None
# NOTES:
# Waveforms are centered by negative or positive peak
# Waveforms in .npy files have shape (num_spikes, num_samples, num_channels)
# num_samples = n_samples_before + 1 (1 for location of spike) + n_samples_after

######################################################
###############  CURATION PARAMETERS  ################
######################################################
# If True, units are curated based on quality of spikes (firing rate, ISI, SNR)
AUTO_CURATE_SPIKES = True

# If True, units are curated based on quality of waveforms (min spikes per unit, max norm std)
AUTO_CURATE_WAVEFORMS = True
# If both AUTO_CURATE_SPIKES and AUTO_CURATE_WAVEFORMS are True, then spike curation precedes waveform curation

# If any is None, that auto-curation is skipped.
# Units that do not meet thresholds are removed.

######################################################
##################  SPIKE CURATION  ##################
######################################################
# Curate units based on the quality of the spikes

# Firing rate
# (smaller values are removed)
FR_MIN = 0.05

# ISI-violation ratio (ratio of violation_rate / total_spike_rate)
# (greater values are removed)
ISI_VIOL_MAX = 0.3

# Signal-to-noise ratio (smaller values are removed)
SNR_MIN = 6  # 5

######################################################
################  WAVEFORM CURATION  #################
######################################################
# Curate units based on the quality of waveforms

# Minimum number of spikes per unit
# (smaller values are removed)
SPIKES_PER_UNIT_MIN = 50

# Maximum normalized standard deviation (standard deviation of waveform divided (normalized) by amplitude)
# (greater values are removed)
STD_NORM_MAX = 0.3
# If True, use the standard deviation at the peak. If False, use the average standard deviation over the waveform window
STD_AT_PEAK = True
# If STD_AT_PEAK = False, the waveform window for the average standard deviation
STD_OVER_WINDOW_MS_BEFORE = 0.5
STD_OVER_WINDOW_MS_AFTER = 1.5

######################################################
################  MATLAB PARAMETERS  #################
######################################################
# Parameters for resulting .mat file that contains sorted output
# If True, include the electrode number [1, 26400] to the "unit" data structure in the sorted .mat file
SAVE_ELECTRODES = True

######################################################
################  FIGURE PARAMETERS  #################
######################################################
"""
If AUTO_CREATE_FIGURES is True, the following figures (each as a separate .png file)
will be created, containing data from all recordings. 
They will be stored in the last folder in RESULTS_FOLDERS

1) Bar plot showing how many units before and after curation based on waveform.
   Will only be created if AUTO_CURATE_WAVEFORMS is True

2) Scatter plot where each dot represents a single unit (same color = from same recording)
   x-axis is number of spikes. y-axis is norm std described in WAVEFORM CURATION 
   Dotted lines showing SPIKES_PER_UNIT_MIN and STD_NORM_MAX will be plotted
   Will only be created if AUTO_CURATE_WAVEFORMS is True and SPIKES_PER_UNIT_MIN is not None 
   and STD_NORM_MAX is not None
   
3) All template (average) waveforms that pass curation based on spikes plotted together, ordered by amplitude (top = highest amplitude)
   Black waveforms indicates that the corresponding unit of the waveform passed all curation thresholds.
   Red waveforms indicates that the unit did not pass all curation thresholds.
   Vertical dotted lines at LINE_MS_BEFORE_PEAK and LINE_MS_AFTER_PEAK will be plotted before and after
   the waveform peaks, respectively
   Will only be created if AUTO_CURATE_SPIKES is True

NOTE: All waveforms must have been extracted with the same parameters
"""
AUTO_CREATE_FIGURES = True


"""
for plotting all waveforms,

for loop with i
y - 10

so that the waveforms are stacked vertically on top of each other (with the highest amplitude waveform being highest on the y-axis)
if too many waveforms, separate plot into different columns
"""

##############
# For plot 1 #
##############
# Label for x-axis
BAR_X_LABEL = "Recording"
# Label for y-axis
BAR_Y_LABEL = "Number of Units"

##############
# For plot 2 #
##############
# The number of units to randomly sample from each recording to plot on the std scaled scatter plot
# If None-> all units are used
SCATTER_STD_MAX_UNITS_PER_RECORDING = None
# The possible color choices of the different recordings in hexadecimal
# If there are fewer recordings than colors, not all colors will be used
# The length of SCATTER_RECORDING_COLORS must be at least the length of RECORDING_FILES
SCATTER_RECORDING_COLORS = [
    "#f74343",  # red
    "#fccd56",  # yellow
    "#74fc56",  # green
    "#56fcf6",  # light blue
    "#1e1efa",  # dark blue
    "#fa1ed2",  # pink
]
# Label for x-axis
SCATTER_X_LABEL = "Number of Spikes"
# Label for y-axis
SCATTER_Y_LABEL = "avg. STD / amplitude"
# Buffer for setting limits of x-axs
# The axis will go from 0 to maximum_number_of_spikes + SCATTER_X_MAX_BUFFER
SCATTER_X_MAX_BUFFER = 300
# Buffer for setting limits of y-axs
# The axis will go from 0 to maximum_std + SCATTER_Y_MAX_BUFFER
SCATTER_Y_MAX_BUFFER = 0.2

# For plot 3
# Vertical dotted lines at LINE_MS_BEFORE_PEAK and LINE_MS_AFTER_PEAK will be plotted before and after
# the waveform peaks, respectively
# If None-> no line is plotted
ALL_TEMPLATES_MS_BEFORE_PEAK = 0.5
ALL_TEMPLATES_MS_AFTER_PEAK = 1.5

# TODO
# 1) Auto create plots
# 2) Reformat waveform files so that they are sorted based on amplitude
# 3) Add try-except catch in case of error with Kilosort when using parallel processing. reconvert recording with different params


######################################################
######################  CODE  ########################
######################################################


assert len(RECORDING_FILES) == len(INTERMEDIATE_FOLDERS) == len(RESULTS_FOLDERS), "'RECORDING_FILES'" \
                                                                                 " 'INTERMEDIATE_FOLDERS' " \
                                                                                 "and 'RESULTS_FOLDERS' " \
                                                                                 "should all have the same length"

if AUTO_CREATE_FIGURES:
    assert len(SCATTER_RECORDING_COLORS) >= len(RECORDING_FILES), "The length of 'SCATTER_RECORDING_COLORS' must be at " \
                                                                 "least the length of 'RECORDING_FILES'"


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
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
import subprocess
import tempfile
import shutil
import signal
from pathlib import Path
import json
from typing import Optional, List, Any, Union
from tqdm import tqdm
from scipy.io import savemat
import pandas as pd
from natsort import natsorted
from collections import OrderedDict
print(f"Done. Time: {time.time()-_import_start:.2f}s")
# endregion


# region Kilosort
class RunKilosort:
    # Must create Kilosort instance to 1) set paths 2) check if kilosort is installed
    def __init__(self):
        # Set paths
        self.path = self.set_kilosort_path(KILOSORT_PATH)
        os.environ['HDF5_PLUGIN_PATH'] = HDF5_PLUGIN_PATH

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
            projection_threshold=KILOSORT_PARAMS['projection_threshold'],
            preclust_threshold=KILOSORT_PARAMS['preclust_threshold'],
            minfr_goodchannels=KILOSORT_PARAMS['minfr_goodchannels'],
            minFR=KILOSORT_PARAMS['minFR'],
            freq_min=KILOSORT_PARAMS['freq_min'],
            sigmaMask=KILOSORT_PARAMS['sigmaMask'],
            kilo_thresh=KILOSORT_PARAMS['detect_threshold'],
            use_car=KILOSORT_PARAMS['car'],
            nPCs=int(KILOSORT_PARAMS['nPCs']),
            ntbuff=int(KILOSORT_PARAMS['ntbuff']),
            nfilt_factor=int(KILOSORT_PARAMS['nfilt_factor']),
            NT=int(KILOSORT_PARAMS['NT'])
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

        if AUTO_SETUP_POD_ENV:
            os.environ['MW_NVCC_PATH'] = "/usr/local/cuda-10.1/bin"
            shell_cmd = f'''
                        #!/bin/bash
                        cd "{output_folder}"
                        module load MatLab/R2021b
                        module load cuda/10.1
                        matlab -nosplash -nodisplay -log -r kilosort2_master
                    '''
        else:
            shell_cmd = f'''
                        #!/bin/bash
                        cd "{output_folder}"
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
        if KILOSORT_PARAMS['NT'] is None:
            KILOSORT_PARAMS['NT'] = 64 * 1024 + KILOSORT_PARAMS['ntbuff']
        else:
            KILOSORT_PARAMS['NT'] = KILOSORT_PARAMS['NT'] // 32 * 32  # make sure is multiple of 32

        if KILOSORT_PARAMS['car']:
            KILOSORT_PARAMS['car'] = 1
        else:
            KILOSORT_PARAMS['car'] = 0

    @classmethod
    def get_result_from_folder(cls, output_folder):
        return KilosortSortingExtractor(folder_path=output_folder)


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

    def __init__(self, folder_path, exclude_cluster_groups=None):
        # Folder containing the numpy results of Kilosort
        phy_folder = Path(folder_path)
        self.folder = phy_folder.absolute()

        self.spike_times = np.atleast_1d(np.load(str(phy_folder / 'spike_times.npy')).astype(int).flatten())
        self.spike_clusters = np.atleast_1d(np.load(str(phy_folder / 'spike_clusters.npy')).flatten())

        # The unit_ids with at least 1 spike
        unit_ids_with_spike = set(self.spike_clusters)

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

        if KILOSORT_PARAMS["keep_good_only"] and "KSLabel" in cluster_info.columns:
            cluster_info = cluster_info.query("KSLabel == 'good'")

        all_unit_ids = cluster_info["cluster_id"].values
        self.unit_ids = []
        # Exclude units with 0 spikes
        for unit_id in all_unit_ids:
            if unit_id in unit_ids_with_spike:
                self.unit_ids.append(int(unit_id))

    @staticmethod
    def get_num_segments():
        # Sorting should always have 1 segment
        return 1

    def get_unit_spike_train(self,
                             unit_id,
                             segment_index: Union[int, None] = None,
                             start_frame: Union[int, None] = None,
                             end_frame: Union[int, None] = None,
                             ):
        spike_times = self.spike_times[self.spike_clusters == unit_id]
        if start_frame is not None:
            spike_times = spike_times[spike_times >= start_frame]
        if end_frame is not None:
            spike_times = spike_times[spike_times < end_frame]

        return np.atleast_1d(spike_times.copy().squeeze())

    def get_templates_all(self):
        # Returns Kilosort2's outputted templates as mmap np.array
        return np.load(str(self.folder / "templates.npy"), mmap_mode="r")

    def get_channel_map(self):
        # Returns Kilosort2's channel map as mmap np.array
        return np.load(str(self.folder / "channel_map.npy"), mmap_mode="r").squeeze()

    def get_chans_max(self):
        """
        Get the max channel of each unit based on Kilosort2's template
        and whether to use (min/argmin or max/argmax) for computing peak values

        Returns
        -------
        All are np.arrays that follow np.array[unit_id] = value
        In other words, the np.arrays contain data for ALL units (even units with 0 spikes)

        use_pos_peak
            0 = Use negative peak
            1 = Use positive peak
        chans_max_kilosort
            The channel with the highest amplitude for each unit based on kilosort's selected channels
            that were used during spike sorting (considered not "bad channels")
        chans_max
            The channel with the highest amplitude for each unit converted from kilosort's channels
            to channels in the recording (with all channels)
        """

        templates_all = self.get_templates_all()

        chans_neg_peaks_values = np.min(templates_all, axis=1)
        chans_neg_peaks_indices = chans_neg_peaks_values.argmin(axis=1)
        chans_neg_peaks_values = np.min(chans_neg_peaks_values, axis=1)

        chans_pos_peaks_values = np.max(templates_all, axis=1)
        chans_pos_peaks_indices = chans_pos_peaks_values.argmax(axis=1)
        chans_pos_peaks_values = np.max(chans_pos_peaks_values, axis=1)

        use_pos_peak = chans_pos_peaks_values >= POS_PEAK_THRESH * np.abs(chans_neg_peaks_values)
        chans_max_kilosort = np.where(use_pos_peak, chans_pos_peaks_indices, chans_neg_peaks_indices)
        chans_max_all = self.get_channel_map()[chans_max_kilosort]

        return use_pos_peak, chans_max_kilosort, chans_max_all

    def get_templates_half_windows_sizes(self, chans_max_kilosort, window_size_scale=0.75):
        """
        Get the half window sizes that will be used to recenter the spike times on the peak

        Parameters
        ----------
        chans_max_kilosort: np.array
            np.array with shape (n_templates,) giving the max channel of each template using
            Kilosort's channel map
        window_size_scale: float
            Value to scale the window size for finding the peak
                Smaller = smaller window, less risk of picking wrong peak, higher risk of picking not the peak value of the peak

        Returns
        -------

        """
        # Get the half window sizes that will be used to recenter the spike times on the peak
        templates_all = self.get_templates_all()[np.arange(chans_max_kilosort.size), :, chans_max_kilosort]
        n_templates, n_samples = templates_all.shape
        template_mid = n_samples//2
        half_windows_sizes = []
        for i in range(n_templates):
            template = templates_all[i, :]
            size = template_mid - np.flatnonzero(np.isclose(template[:template_mid], 0))[-1]
            half_windows_sizes.append(int(size*window_size_scale))

        return half_windows_sizes


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
# endregion


# region Extract Waveforms
class WaveformExtractor:
    # region Initialize
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
        self.nbefore = self.ms_to_samples(WAVEFORMS_MS_BEFORE)  # Number of samples before waveform peak to include
        self.nafter = self.ms_to_samples(WAVEFORMS_MS_AFTER) + 1    # Number of samples after waveform peak to include (+1 since Python slicing is [inlusive, exclusive))
        self.nsamples = self.nbefore + self.nafter  # Total number of samples in waveform

        return_scaled_corrected = RETURN_SCALED
        if return_scaled_corrected:
            # Check if recording has scaled values:
            if not self.recording.has_scaled_traces():
                print("Setting 'RETURN_SCALED' to False")
                return_scaled_corrected = False
        self.return_scaled = return_scaled_corrected

        dtype = WAVEFORM_DTYPE
        if dtype is None:
            dtype = self.recording.get_dtype()
        if np.issubdtype(dtype, np.integer) and return_scaled_corrected:
            # If waveform will be scaled, dtype cannot be an integer (that would cause rounding)
            print("Setting 'dtype' to float32 because 'RETURN_SCALED' is True")
            dtype = "float32"
        dtype = np.dtype(dtype)
        self.dtype = dtype.str

    @classmethod
    def create(cls, recording, sorting, folder):
        # Create new waveform extractor
        folder = Path(folder)
        create_folder(folder / 'waveforms')
        print()
        we = cls(recording, sorting, folder)

        # Get template window sizes for computing location of negative peak during waveform extraction
        we.use_pos_peak, we.chans_max_kilosort, we.chans_max_all = we.sorting.get_chans_max()
        we.templates_half_windows_sizes = we.sorting.get_templates_half_windows_sizes(we.chans_max_kilosort)

        all_waveforms_folder = folder.parent
        for save_file, save_data in zip(("use_pos_peak.npy", "chans_max_kilosort.npy", "chans_max_all.npy"),
                                        (we.use_pos_peak, we.chans_max_kilosort, we.chans_max_all)):
            np.save(all_waveforms_folder / save_file, save_data)
        np.save(all_waveforms_folder / "peak_ind.npy", we.nbefore)

        np.save(str(folder / "unit_ids.npy"), we.sorting.unit_ids)

        return we

    @classmethod
    def load_from_folder(cls, recording, sorting, folder,
                         use_pos_peak=None, chans_max_kilosort=None, chans_max_all=None):
        # Load waveform data from folder
        we = cls(recording, sorting, folder)

        _possible_template_modes = ('average', 'std', 'median')
        for mode in _possible_template_modes:
            # Load cached templates
            template_file = folder / f'templates_{mode}.npy'
            if template_file.is_file():
                we.template_cache[mode] = np.load(template_file, mmap_mode="r")

        if use_pos_peak is None:
            all_waveforms_folder = we.folder.parent
            we.use_pos_peak = np.load(all_waveforms_folder / "use_pos_peak.npy", mmap_mode="r")
            we.chans_max_kilosort = np.load(all_waveforms_folder / "chans_max_kilosort.npy", mmap_mode="r")
            we.chans_max_all = np.load(all_waveforms_folder / "chans_max_all.npy", mmap_mode="r")
        else:
            we.use_pos_peak = use_pos_peak
            we.chans_max_kilosort = chans_max_kilosort
            we.chans_max_all = chans_max_all

        we.load_unit_ids()
        return we

    def ms_to_samples(self, ms):
        return int(ms * self.sampling_frequency / 1000.)

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
                spike_times_sel = spike_times[sel]

                selected_spike_times[unit_id].append(spike_times_sel)

        # Prepare memmap for waveforms
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

        # Run extract waveforms
        func = WaveformExtractor._waveform_extractor_chunk
        init_func = WaveformExtractor._init_worker_waveform_extractor

        init_args = (self.recording, self.sorting, self, wfs_memmap, selected_spikes, selected_spike_times, self.nbefore, self.nafter, self.return_scaled)
        processor = ChunkRecordingExecutor(self.recording, func, init_func, init_args, job_name='extract waveforms',
                                           handle_returns=True, **job_kwargs)
        spike_times_centered_dicts = processor.run()

        # Copy original kilosort spike times
        shutil.copyfile(self.sorting.folder / "spike_times.npy", self.sorting.folder / "spike_times_kilosort.npy")

        # Center spike times
        spike_times = self.sorting.spike_times
        spike_time_to_ind = {}
        for i, st in enumerate(spike_times):
            spike_time_to_ind[st] = i

        for st_dict in spike_times_centered_dicts:
            for st, st_cen in st_dict.items():
                spike_times[spike_time_to_ind[st]] = st_cen
        np.save(self.sorting.folder / "spike_times.npy", spike_times)

    def sample_spikes(self):
        """
        Uniform random selection of spikes per unit and save to .npy

        self.samples_spikes just calls self.random_spikes_uniformly and saves data to .npy files

        Returns
        -------
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
            if MAX_WAVEFORMS_PER_UNIT is not None:
                if total > MAX_WAVEFORMS_PER_UNIT:
                    global_inds = np.random.choice(total, size=MAX_WAVEFORMS_PER_UNIT, replace=False)
                    global_inds = np.sort(global_inds)
                else:
                    global_inds = np.arange(total)
            else:
                global_inds = np.arange(total)
            sel_spikes = []
            for segment_index in range(num_seg):
                in_segment = (global_inds >= cum_sum[segment_index]) & (global_inds < cum_sum[segment_index + 1])
                inds = global_inds[in_segment] - cum_sum[segment_index]

                if MAX_WAVEFORMS_PER_UNIT is not None:
                    # clean border when sub selection
                    assert self.nafter is not None
                    spike_times = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                    sampled_spike_times = spike_times[inds]
                    num_samples = self.recording.get_num_samples(segment_index=segment_index)
                    mask = (sampled_spike_times >= self.nbefore) & (sampled_spike_times < (num_samples - self.nafter))
                    inds = inds[mask]

                sel_spikes.append(inds)
            selected_spikes[unit_id] = sel_spikes
        return selected_spikes

    @staticmethod
    def _waveform_extractor_chunk(segment_index, start_frame, end_frame, worker_ctx):
        # recover variables of the worker
        recording = worker_ctx['recording']
        sorting = worker_ctx['sorting']

        waveform_extractor = worker_ctx['waveform_extractor']
        templates_half_windows_sizes = waveform_extractor.templates_half_windows_sizes
        use_pos_peak = waveform_extractor.use_pos_peak
        chans_max_all = waveform_extractor.chans_max_all

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

        spike_times_centered = {}
        if len(to_extract) > 0:
            start = min(st[0]-nbefore-templates_half_windows_sizes[uid] for uid, (_, _, st) in to_extract.items())   # Get the minimum time frame from recording needed for extracting waveform from the minimum spike time - nbefore
            end = max(st[-1]+nbefore+templates_half_windows_sizes[uid] for uid, (_, _, st) in to_extract.items())
            start = int(start)
            end = int(end)

            # load trace in memory
            traces = recording.get_traces(start_frame=start, end_frame=end, segment_index=segment_index,
                                          return_scaled=return_scaled)
            max_trace_ind = traces.shape[0] - 1
            for unit_id, (i0, i1, local_spike_times) in to_extract.items():
                wfs = np.load(wfs_memmap_files[unit_id], mmap_mode="r+")
                half_window_size = templates_half_windows_sizes[unit_id]
                chan_max = chans_max_all[unit_id]
                for i in range(local_spike_times.size):
                    st = int(local_spike_times[i])  # spike time
                    st_trace = st - start  # Convert the spike time defined by all the samples in recording to only samples in "traces"

                    peak_window_left = max(st_trace - half_window_size, 0)
                    peak_window_right = min(st_trace + half_window_size, max_trace_ind)
                    peak_window_size = peak_window_right - peak_window_left + 1
                    traces_peak_window = traces[peak_window_left:peak_window_right, chan_max]
                    if use_pos_peak[unit_id]:
                        peak_value = np.max(traces_peak_window)
                    else:
                        peak_value = np.min(traces_peak_window)
                    peak_indices = np.flatnonzero(traces_peak_window == peak_value)
                    st_offset = peak_indices[peak_indices.size//2] - peak_window_size//2
                    st_trace += st_offset
                    spike_times_centered[st] = st + st_offset

                    pos = unit_cum_sum[unit_id][segment_index] + i0 + i  # Index for waveform along 0th axis in .npy waveforms file
                    wfs[pos, :, :] = traces[st_trace - nbefore:st_trace + nafter, :]  # Python slices with [start, end), so waveform is in format (nbefore + spike_location + nafter-1, n_channels)
        return spike_times_centered

    @staticmethod
    def _init_worker_waveform_extractor(recording, sorting, waveform_extractor, wfs_memmap,
                                        selected_spikes, selected_spike_times, nbefore, nafter, return_scaled):
        # create a local dict per worker
        worker_ctx = {}
        worker_ctx['recording'] = recording
        worker_ctx['sorting'] = sorting
        worker_ctx['waveform_extractor'] = waveform_extractor

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
    def get_waveforms(self, unit_id, with_index=False, cache=False, memmap=True):  # SpikeInterface has cache=True by default
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
            num_samples = nbefore + 1 (for value at peak) + nafter
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

    def get_computed_template(self, unit_id, mode):
        """
        Return template (average waveform).

        Parameters
        ----------
        unit_id: int
            Unit id to retrieve waveforms for
        mode: str
            'average' (default), 'median' , 'std'(standard deviation)
        Returns
        -------
        template: np.array
            The returned template (num_samples, num_channels)
        """

        _possible_template_modes = {'average', 'std', 'median'}
        assert mode in _possible_template_modes
        # assert unit_id in self.sorting.unit_ids

        if mode in self.template_cache:
            # already in the global cache
            template = self.template_cache[mode][unit_id, :, :]
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

    def compute_templates(self, modes=('average', 'std'), unit_ids=None, folder=None):
        """
        Compute all template for different "modes":
          * average
          * std
          * median

        The results are cached in memory as 3d ndarray (nunits, nsamples, nchans)
        and also saved as npy file in the folder to avoid recomputation each time.

        Parameters
        ----------
        modes: tuple
            Template modes to compute (average, std, median)
        unit_ids: None or List
            Unit ids to compute templates for
            If None-> unit ids are taken from self.sorting.unit_ids
        folder: None or Path
            Folder to save templates to
            If None-> use self.folder
        """
        # TODO : run this in parralel

        print_stage("COMPUTING TEMPLATES")
        print("Template modes: " + ", ".join(modes))
        stopwatch = Stopwatch()

        if unit_ids is None:
            unit_ids = self.sorting.unit_ids
        if folder is None:
            folder = self.folder

        num_chans = self.recording.get_num_channels()

        for mode in modes:
            # With max(unit_ids)+1 instead of len(unit_ids), the template of unit_id can be retrieved by template[unit_id]
            # Instead of first converting unit_id to an index
            templates = np.zeros((max(unit_ids)+1, self.nsamples, num_chans), dtype=self.dtype)
            self.template_cache[mode] = templates

        for unit_id in unit_ids:
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

                self.template_cache[mode][unit_id, :, :] = arr

        print("Saving templates to .npy")
        for mode in modes:
            templates = self.template_cache[mode]
            template_file = folder / f'templates_{mode}.npy'
            np.save(str(template_file), templates)
        stopwatch.log_time("Done computing and saving templates.")

    def load_unit_ids(self):
        unit_ids_folder = self.folder / "unit_ids.npy"

        if unit_ids_folder.exists():
            self.sorting.unit_ids = np.load(self.folder / "unit_ids.npy")

    def get_curation_history(self):
        with open(self.folder / "curation_history.json", "r") as f:
            return json.load(f)

    # endregion

    # region Format files
    def save_curated_units(self, curated_folder, curation_history):
        """
        Filters units by storing curated units in a new folder.

        Parameters
        ----------
        curated_folder: Path
            The new folder where curated waveforms are copied
        curation_history: dict
            Contains curation history to be saved

        Return
        ------
        we :  WaveformExtractor
            The newly create waveform extractor with the selected units
        """
        print_stage("SAVING CURATED UNITS")
        stopwatch = Stopwatch()
        unit_ids = self.sorting.unit_ids
        print(f"Saving {len(unit_ids)} curated units to new folder")
        create_folder(curated_folder)

        # Save unit_ids
        np.save(str(curated_folder/"unit_ids.npy"), unit_ids)

        # Create waveforms folder
        curated_waveforms_folder = curated_folder / "waveforms"
        create_folder(curated_waveforms_folder)

        # Copy .npy waveform files to new folder
        waveforms_files = [f for f in (self.folder / "waveforms").iterdir() if f.suffix == ".npy"]
        for unit in unit_ids:
            for wf_file in waveforms_files:
                if f"waveforms_{unit}.npy" in wf_file.name or f'sampled_index_{unit}.npy' in wf_file.name:
                    shutil.copyfile(wf_file, curated_waveforms_folder / wf_file.name)

        # Copy computed templates (will contain templates even from units that did not pass curation)
        for file in self.folder.iterdir():
            if file.name.startswith("templates_"):
                shutil.copyfile(file, curated_folder / file.name)

        we = WaveformExtractor.load_from_folder(self.recording, self.sorting, curated_folder,
                                                self.use_pos_peak, self.chans_max_kilosort, self.chans_max_all)
        stopwatch.log_time("Done saving curated units.")

        # Save curation history
        print_stage("SAVING CURATION HISTORY")
        stopwatch = Stopwatch()
        with open(curated_folder / "curation_history.json", "w") as f:
            json.dump(curation_history, f)
        stopwatch.log_time("Done.")

        return we
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
    def update_history(curation_history, key, curated, failed, metrics):
        curation_history["curations"].append(key)
        curation_history["curated"][key] = curated
        curation_history["failed"][key] = failed
        curation_history["metrics"][key] = metrics

    @staticmethod
    def spikes_min(sorting):
        """
        Curate units in sorting.unit_ids based on minimum number of spikes per unit
        Units with greater spikes than SPIKES_PER_UNIT_MIN are curated

        Parameters
        ----------
        sorting: KilosortSortingExtractor
            Must have sorting.unit_ids and sorting.get_unit_spike_train(unit_id=)

        Returns
        -------
        curated_unit_ids: list
            Contains the unit_ids that passed the curation
        failed_unit_ids: list
            Contains the unit_ids that did not pass the curation
        metrics: dict
            Contains the curation metric of each unit
            metrics[unit_id] = value
        """
        curated_unit_ids = []
        failed_unit_ids = []
        metrics = {}

        for unit_id in sorting.unit_ids:
            num_spikes = sorting.get_unit_spike_train(unit_id=unit_id).size
            if num_spikes >= SPIKES_PER_UNIT_MIN:
                curated_unit_ids.append(unit_id)
            else:
                failed_unit_ids.append(unit_id)
            metrics[unit_id] = float(num_spikes)

        return curated_unit_ids, failed_unit_ids, metrics

    @staticmethod
    def firing_rate(recording, sorting):
        """
        Curate units in sorting.unit_ids based on firing rate
        Units with firing rate greater than FR_MIN are curated

        Parameters
        ----------
        recording
            Must have recording.get_num_samples() and recording.get_sampling_frequency()
        sorting: KilosortSortingExtractor
            Must have sorting.unit_ids and sorting.get_unit_spike_train(unit_id=)

        Returns
        -------
        curated_unit_ids: list
            Contains the unit_ids that passed the curation
        failed_unit_ids: list
            Contains the unit_ids that did not pass the curation
        metrics: dict
            Contains the curation metric of each unit
            metrics[unit_id] = value
        """

        total_duration = recording.get_num_samples() / recording.get_sampling_frequency()

        curated_unit_ids = []
        failed_unit_ids = []
        metrics = dict()

        for unit_id in sorting.unit_ids:
            firing_rate = sorting.get_unit_spike_train(unit_id=unit_id).size / total_duration
            if firing_rate >= FR_MIN:
                curated_unit_ids.append(unit_id)
            else:
                failed_unit_ids.append(unit_id)
            metrics[unit_id] = float(firing_rate)

        return curated_unit_ids, failed_unit_ids, metrics

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
        curated_unit_ids: list
            Contains the unit_ids that passed the curation
        failed_unit_ids: list
            Contains the unit_ids that did not pass the curation
        metrics: dict
            Contains the curation metric of each unit
            metrics["unit_id"] = value

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
        sampling_frequency = recording.get_sampling_frequency()
        total_duration = recording.get_num_samples() / sampling_frequency

        # All units converted to seconds
        isi_threshold_s = isi_threshold_ms / 1000
        min_isi_s = min_isi_ms / 1000
        isi_threshold_samples = int(isi_threshold_s * sampling_frequency)

        curated_unit_ids = []
        failed_unit_ids = []
        metrics = dict()
        for unit_id in sorting.unit_ids:
            spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
            num_spikes = spike_train.size
            isis = np.diff(spike_train)

            violation_num = np.sum(isis < isi_threshold_samples)
            violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s)

            total_rate = num_spikes / total_duration
            violation_rate = violation_num / violation_time
            violation_rate_ratio = violation_rate / total_rate

            if violation_rate_ratio <= ISI_VIOL_MAX:
                curated_unit_ids.append(unit_id)
            else:
                failed_unit_ids.append(unit_id)
            metrics[unit_id] = float(violation_rate_ratio)
        return curated_unit_ids, failed_unit_ids, metrics

    @staticmethod
    def std_norm_max(waveform_extractor):
        """
        Curate units in waveform_extractor.sorting.unit_ids based on maximum normalized standard deviation
        Maximum normalized standard deviation is based on
        standard deviation at peak or over entire waveform window (determined by user parameters)
        divided (normalized) by amplitude since higher amplitude units will have greater absolute std

        Units with maximum normalized standard deviation below STD_NORM_MAX are curated

        Parameters
        ----------
        waveform_extractor: WaveformExtractor

        Returns
        -------
        curated_unit_ids: list
            Contains the unit_ids that passed the curation
        failed_unit_ids: list
            Contains the unit_ids that did not pass the curation
        metrics: dict
            Contains the curation metric of each unit
            metrics["unit_id"] = value
        """
        curated_unit_ids = []
        failed_unit_ids = []
        metrics = dict()

        unit_ids = waveform_extractor.sorting.unit_ids
        print(f"Iterating through {len(unit_ids)} units")
        for unit_id in tqdm(unit_ids):
            chan_max = waveform_extractor.chans_max_all[unit_id]
            peak_ind = waveform_extractor.nbefore

            amp_avg = waveform_extractor.get_computed_template(unit_id=unit_id, mode="average")[peak_ind, chan_max]
            template_std = waveform_extractor.get_computed_template(unit_id=unit_id, mode="std")[:, chan_max]

            if STD_AT_PEAK:
                std = template_std[peak_ind]
            else:
                nbefore = waveform_extractor.ms_to_samples(STD_OVER_WINDOW_MS_BEFORE)
                nafter = waveform_extractor.ms_to_samples(STD_OVER_WINDOW_MS_AFTER) + 1  # +1 since end of slice is exclusive

                std = np.mean(template_std[peak_ind-nbefore:peak_ind+nafter])

            std_norm = std / amp_avg
            if std_norm <= STD_NORM_MAX:
                curated_unit_ids.append(unit_id)
            else:
                failed_unit_ids.append(unit_id)
            metrics[unit_id] = float(std_norm)

        return curated_unit_ids, failed_unit_ids, metrics

    @staticmethod
    def snr(waveform_extractor):
        """
        Curate units in waveform_extractor.sorting.unit_ids based on
        ratio of peak amplitude to noise of the channel that defines the peak amplitude

        Parameters
        ----------
        waveform_extractor: WaveformExtractorl

        Returns
        -------
        curated_unit_ids: list
            Contains the unit_ids that passed the curation
        failed_unit_ids: list
            Contains the unit_ids that did not pass the curation
        metrics: dict
            Contains the curation metric of each unit
            metrics[unit_id] = value
        """
        recording = waveform_extractor.recording
        sorting = waveform_extractor.sorting

        noise_levels = Curation.get_noise_levels(recording, waveform_extractor.return_scaled)

        curated_unit_ids = []
        failed_unit_ids = []
        metrics = dict()
        for unit_id in sorting.unit_ids:
            chan_ind = waveform_extractor.chans_max_all[unit_id]
            template = waveform_extractor.get_computed_template(unit_id, mode="average")

            peak_ind = waveform_extractor.nbefore
            amplitude = np.abs(template[peak_ind, chan_ind])
            noise = noise_levels[chan_ind]
            with np.errstate(divide='ignore'):  # Ignore divide by zero warnings
                snr = amplitude / noise
                if snr >= SNR_MIN:
                    curated_unit_ids.append(unit_id)
                else:
                    failed_unit_ids.append(unit_id)
                metrics[unit_id] = float(snr)

        return curated_unit_ids, failed_unit_ids, metrics

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
# endregion


# region spikesort_matlab.py
class Stopwatch:
    def __init__(self, start_msg=None, use_print_stage=True):
        if start_msg is not None:
            if use_print_stage:
                print_stage(start_msg)
            else:
                print(start_msg)

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


def load_raw_recording(rec_path):
    print_stage("LOADING RAW MAXWELL RECORDING")
    print(f"Recording path: {rec_path}")
    try:
        stopwatch = Stopwatch()
        if str(rec_path).endswith(".raw.h5"):
            rec = MaxwellRecordingExtractor(rec_path)
        else:
            rec = concatenate_recordings(rec_path)

        assert rec.get_num_segments() == 1, "Recording has multiple segments. Divide recording into multiple recordings"

        print('Flipping y-coordinates of channel locations ')
        probes_all = []
        for probe in rec.get_probes():
            y_cords = probe._contact_positions[:, 1]
            y_cords_flipped = max(y_cords) - y_cords
            probe._contact_positions[np.arange(y_cords_flipped.size), 1] = y_cords_flipped
            probes_all.append(probe)
        rec = rec.set_probes(probes_all)

        stopwatch.log_time("Done loading recording.")
        return rec
    except Exception as e:
        print(f"Could not open the recording file because of {e}")
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
    print(f"Total duration: {rec.get_total_duration()}s")
    return rec


def get_paths(rec_path, inter_path):

    rec_path = Path(rec_path)
    rec_name = rec_path.name.split(".")[0]
    print_stage("PROCESSING RECORDING")
    print(f"Recording path: {rec_path}")

    inter_path = Path(inter_path)

    recording_dat_path = inter_path / (rec_name + "_filtered.dat")
    recording_saved_path = inter_path / (rec_name + "_filtered")
    output_folder = inter_path / 'kilosort2_results'
    waveforms_folder = inter_path / 'waveforms'
    waveforms_raw_folder = waveforms_folder / 'waveforms_raw'
    waveforms_curated_spikes_folder = waveforms_folder / 'waveforms_curated_spikes'
    waveforms_curated_waveforms_folder = waveforms_folder / 'waveforms_curated_waveforms'

    delete_folders = []
    if RECOMPUTE_RECORDING:
        delete_folders.append(inter_path)
    if RECOMPUTE_SORTING:
        delete_folders.extend((output_folder, waveforms_folder))
    if REEXTRACT_WAVEFORMS:
        delete_folders.append(waveforms_folder)
    if RECURATE_SPIKES:
        delete_folders.append(waveforms_curated_spikes_folder)
    if RECURATE_WAVEFORMS:
        delete_folders.append(waveforms_curated_waveforms_folder)
    for folder in delete_folders:
        if folder.exists():
            shutil.rmtree(folder)
            print(f"Deleted folder: {folder}")

    create_folder(inter_path)
    return rec_path, inter_path, recording_saved_path, recording_dat_path, output_folder, waveforms_raw_folder, waveforms_curated_spikes_folder, waveforms_curated_waveforms_folder


def write_recording(recording, recording_saved_path, recording_dat_path, verbose=True):
    stopwatch = Stopwatch(start_msg="WRITING RECORDING", use_print_stage=True)
    job_kwargs = {"progress_bar": verbose, "verbose": verbose, "n_jobs": N_JOBS, "total_memory": TOTAL_MEMORY}

    if not (recording_saved_path / "cached.json").exists():  # Last file to be created when saving
        print(f"Computing bandpass filter and saving recording to {recording_saved_path}")
        recording_filtered = bandpass_filter(recording, freq_min=FREQ_MIN, freq_max=FREQ_MAX)
        # More noise with float32
        # recording_filtered = bandpass_filter(recording, freq_min=FREQ_MIN, freq_max=FREQ_MAX, dtype="float32")

        print_stage("FIX ME")
        recording_filtered_saved = recording_filtered
        # if recording_saved_path.exists():  # If already partially saved, delete file (SpikeInterface expects new folder)
        #     shutil.rmtree(recording_saved_path)
        # recording_filtered_saved = recording_filtered.save(folder=recording_saved_path, **job_kwargs)
    else:
        print(f"Loading existing saved bandpass-filtered recording at {recording_saved_path}")
        recording_filtered_saved = MaxwellRecordingExtractor.load(recording_saved_path)

    print(f"\nKilosort2's .dat path: {recording_dat_path}")
    if not recording_dat_path.exists():
        # dtype has to be 'int16' (that's what Kilosort2 expects--but can change)
        print("Converting .raw.h5 recording to .dat format for Kilosort2")
        BinaryRecordingExtractor.write_recording(recording_filtered_saved, file_paths=recording_dat_path,
                                                 dtype='int16', **job_kwargs)
    else:
        print(f"Using existing .dat as recording file for Kilosort2")
    stopwatch.log_time("Done writing recording.")
    return recording_filtered_saved


def spike_sort(rec_cache, rec_path, recording_dat_path, output_folder):
    print_stage("SPIKE SORTING")
    stopwatch = Stopwatch()
    sorting = -1

    kilosort = RunKilosort()
    try:
        if not RECOMPUTE_SORTING and (output_folder / "spike_times.npy").exists():
            print("Loading Kilosort2's sorting results")
            sorting = kilosort.get_result_from_folder(output_folder)
        else:
            create_folder(output_folder)
            sorting = kilosort.run(recording=rec_cache, recording_dat_path=recording_dat_path,
                                   output_folder=output_folder)

    except Exception as e:
        print(f"Kilosort2 failed on recording {rec_path} because of {e}")

    stopwatch.log_time("Done sorting.")
    return sorting


def extract_waveforms(recording, sorting, folder, **job_kwargs):
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

    if not REEXTRACT_WAVEFORMS and (folder / "waveforms").is_dir():  # Load saved waveform extractor
        print("Loading waveforms from folder")
        we = WaveformExtractor.load_from_folder(recording, sorting, folder)
        stopwatch.log_time("Done extracting waveforms.")
    else:  # Create new waveform extractor
        create_folder(folder)
        we = WaveformExtractor.create(recording, sorting, folder)
        we.run_extract_waveforms(**job_kwargs)
        stopwatch.log_time("Done extracting waveforms.")

        we.compute_templates(modes=('average', 'std'), folder=we.folder.parent)
    return we


def curate_data(we_raw, waveforms_curated_spikes_folder, waveforms_curated_waveforms_folder):
    curation_history = {
        "curate_spikes": AUTO_CURATE_SPIKES,
        "curate_waveforms": AUTO_CURATE_WAVEFORMS,
        "curation_parameters": {
            "fr_min": FR_MIN,
            "isi_viol_max": ISI_VIOL_MAX,
            "snr_min": SNR_MIN,
            "spikes_per_unit_min": SPIKES_PER_UNIT_MIN,
            "std_norm_max": STD_NORM_MAX,
            "std_at_peak": STD_AT_PEAK,
            "std_over_window_ms_before": STD_OVER_WINDOW_MS_BEFORE,
            "std_over_window_ms_after": STD_OVER_WINDOW_MS_AFTER
        },
        "initial": we_raw.sorting.unit_ids,
        "curations": [],
        "curated": dict(),
        "failed": dict(),
        "metrics": dict(),
    }

    we_curated_spikes, curation_history = curate_spikes(we_raw, curation_history, waveforms_curated_spikes_folder)
    we_curated_waveforms = curate_waveforms(we_curated_spikes, curation_history, waveforms_curated_waveforms_folder)
    return we_curated_waveforms


def curate_spikes(we_raw, curation_history, curated_spikes_folder):
    print_stage("CURATING SPIKES")
    total_stopwatch = Stopwatch()

    recording = we_raw.recording
    sorting = we_raw.sorting

    curation_history_path = curated_spikes_folder / 'curation_history.json'

    if not AUTO_CURATE_SPIKES:
        print("'AUTO_CURATE_SPIKES' is set to False, so skipping spike curation.")
        return we_raw, curation_history

    if not RECURATE_SPIKES and curation_history_path.exists():
        print("Skipping spike curation since already curated")
        print("Loading saved curated data")
        we_curated = WaveformExtractor.load_from_folder(recording, sorting, curated_spikes_folder,
                                                        we_raw.use_pos_peak, we_raw.chans_max_kilosort, we_raw.chans_max_all)
        with open(curation_history_path, "r") as f:
            curation_history = json.load(f)
        total_stopwatch.log_time("Done loading.")
        return we_curated, curation_history

    # Perform curation
    n_units_total = len(sorting.unit_ids)
    print(f"Computing curation metrics on {n_units_total} units")

    # Firing rate
    if FR_MIN is not None:
        stopwatch = Stopwatch("COMPUTING FIRING RATES")

        curated_fr, failed_fr, metrics_fr = Curation.firing_rate(recording, sorting)
        stopwatch.log_time("Done.")
        print(f'{len(curated_fr)}/{n_units_total} units have firing rates >= {FR_MIN}')

        Curation.update_history(curation_history, "fr", curated_fr, failed_fr, metrics_fr)

    # Interspike interval
    if ISI_VIOL_MAX is not None:
        stopwatch = Stopwatch("COMPUTING ISI")

        curated_isi, failed_isi, metrics_isi = Curation.isi_violation(recording, sorting)
        stopwatch.log_time("Done.")
        print(f"{len(curated_isi)}/{n_units_total} units have ISI violations ratios <= {ISI_VIOL_MAX}")

        Curation.update_history(curation_history, "isi", curated_isi, failed_isi, metrics_isi)

    # Signal-to-noise ratio
    if SNR_MIN is not None:
        stopwatch = Stopwatch("COMPUTING SNR")

        curated_snr, failed_snr, metrics_snr = Curation.snr(we_raw)
        stopwatch.log_time("Done.")
        print(f"{len(curated_snr)}/{n_units_total} units have SNR >= {SNR_MIN}")

        Curation.update_history(curation_history, "snr", curated_snr, failed_snr, metrics_snr)

    print_stage("CURATING SPIKES")
    unit_ids_curated = set(sorting.unit_ids)
    print(f'N units before spike curation: {len(unit_ids_curated)}')

    if FR_MIN is not None:
        unit_ids_curated.intersection_update(curated_fr)
        print(f"N units after firing rate curation: {len(unit_ids_curated)}")
    if ISI_VIOL_MAX is not None:
        unit_ids_curated.intersection_update(curated_isi)
        print(f"N units after ISI curation: {len(unit_ids_curated)}")
    if SNR_MIN is not None:
        unit_ids_curated.intersection_update(curated_snr)
        print(f'N units after SNR curation: {len(unit_ids_curated)}')

    # Save only the curated waveforms to a new folder
    sorting.unit_ids = list(unit_ids_curated)
    we_curated = we_raw.save_curated_units(curated_folder=curated_spikes_folder, curation_history=curation_history)
    return we_curated, curation_history


def curate_waveforms(we_raw, curation_history, curated_waveforms_folder):
    print_stage("CURATING WAVEFORMS")
    total_stopwatch = Stopwatch()

    recording = we_raw.recording
    sorting = we_raw.sorting

    curation_history_path = curated_waveforms_folder / 'curation_history.json'

    if not AUTO_CURATE_WAVEFORMS:
        print("'AUTO_CURATE_WAVEFORMS' is set to False, so skipping spike curation.")
        return we_raw, curation_history

    if not RECURATE_WAVEFORMS and curation_history_path.exists():
        print("Skipping waveform curation since already curated")
        print("Loading saved curated data")
        we_curated = WaveformExtractor.load_from_folder(recording, sorting, curated_waveforms_folder,
                                                        we_raw.use_pos_peak, we_raw.chans_max_kilosort, we_raw.chans_max_all)
        with open(curation_history_path, "r") as f:
            curation_history = json.load(f)
        total_stopwatch.log_time("Done loading.")
        return we_curated, curation_history

    # Perform curation
    n_units_total = len(sorting.unit_ids)
    print(f'Computing curation metrics on {n_units_total} units')

    # Minimum spikes per unit
    if SPIKES_PER_UNIT_MIN is not None:
        stopwatch = Stopwatch("COMPUTING MIN SPIKES PER UNIT")

        curation_spikes_min = Curation.spikes_min(sorting)
        stopwatch.log_time("Done.")
        print(f"{len(curation_spikes_min[0])}/{n_units_total} units have >= {SPIKES_PER_UNIT_MIN} spikes")

        Curation.update_history(curation_history, "spikes_min", *curation_spikes_min)

    # Maximum normalized standard deviation
    if STD_NORM_MAX is not None:
        stopwatch = Stopwatch("COMPUTING NORM STD")

        curation_std_norm_max = Curation.std_norm_max(we_raw)
        stopwatch.log_time("Done.")
        print(f"{len(curation_std_norm_max[0])}/{n_units_total} units have norm std <= {STD_NORM_MAX}")

        Curation.update_history(curation_history, "std_norm_max", *curation_std_norm_max)

    print_stage("CURATING WAVEFORMS")
    unit_ids_curated = set(sorting.unit_ids)
    print(f'N units before spike curation: {len(unit_ids_curated)}')

    if SPIKES_PER_UNIT_MIN is not None:
        unit_ids_curated.intersection_update(curation_spikes_min[0])
        print(f"N units after min spikes per unit curation: {len(unit_ids_curated)}")
    if STD_NORM_MAX is not None:
        unit_ids_curated.intersection_update(curation_std_norm_max[0])
        print(f"N units after norm std curation: {len(unit_ids_curated)}")

    # Save only the curated waveforms to a new folder
    sorting.unit_ids = [int(u) for u in unit_ids_curated]
    we_curated = we_raw.save_curated_units(curated_folder=curated_waveforms_folder, curation_history=curation_history)
    return we_curated


def process_recording(rec_path, inter_path):
    # Start time for entire process
    stopwatch = Stopwatch()

    # Get Paths
    rec_path, inter_path, recording_saved_path, recording_dat_path, output_folder, waveforms_raw_folder, waveforms_curated_spikes_folder, waveforms_curated_waveforms_folder = get_paths(rec_path, inter_path)

    # Save a copy of the script
    if SAVE_SCRIPT:
        print_stage("SAVING SCRIPT")
        copy_script(inter_path)

    # Load Recording
    recording_raw = load_raw_recording(rec_path)
    if recording_raw == -1:  # Could not load recording
        return "Could not load recording. Moving on to next recording."

    # Filter and writing .dat for input to Kilosort2
    recording_filtered = write_recording(recording_raw, recording_saved_path, recording_dat_path, verbose=True)

    # print_stage("Test")
    # Curation.get_noise_levels(recording_filtered, return_scaled=True)
    # exit()

    # Spike sorting
    sorting = spike_sort(rec_cache=recording_filtered, rec_path=rec_path, recording_dat_path=recording_dat_path,
                         output_folder=output_folder)
    if sorting == -1:  # Could not sort recording
        return "Could not sort recording. Moving on to next recording"

    # Extract waveforms
    we_raw = extract_waveforms(recording_filtered, sorting, folder=waveforms_raw_folder,
                               n_jobs=N_JOBS, total_memory=TOTAL_MEMORY, progress_bar=True)
    # Curating data
    waveform_extractor = curate_data(we_raw, waveforms_curated_spikes_folder, waveforms_curated_waveforms_folder)

    print_stage("DONE EXTRACTING DATA FROM RECORDING")
    stopwatch.log_time("Total")

    return waveform_extractor


def convert_to_matlab(waveform_extractor, rec_name, results_path):
    print_stage("CONVERTING TO MATLAB")
    stopwatch = Stopwatch()

    # Get matlab file path
    name = rec_name + "_sorted.mat"
    mat_file = Path(results_path) / name

    if mat_file.exists():
        if OVERRIDE_MATLAB:
            mat_file.unlink()
            print(f"Deleted {mat_file}")
        else:
            print(f"Skipping creating {mat_file} since already created")
            return

    # Make parent file
    create_folder(mat_file.parent)

    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting
    mdict = {"units": [], "locations": recording.get_channel_locations(), "fs": recording.get_sampling_frequency()}

    # Get channel locations
    locations = recording.get_channel_locations()

    # Get electrodes
    if SAVE_ELECTRODES:
        electrodes = recording.get_property('electrode')

    for u in sorting.unit_ids:
        chan_max_ind = waveform_extractor.chans_max_all[u]
        location_max = locations[chan_max_ind]

        unit_dict = {
            "unit_id": u,
            "spike_train": sorting.get_unit_spike_train(u),
            "x_max": location_max[0],
            "y_max": location_max[1],
            "template": waveform_extractor.get_computed_template(u, mode="average"),
        }

        if SAVE_ELECTRODES:
            electrode = electrodes[chan_max_ind]
            unit_dict["electrode"] = electrode

        mdict["units"].append(unit_dict)

    savemat(mat_file, mdict)
    print(f"Saved to {mat_file}")
    stopwatch.log_time()

    return mdict


def copy_script(path):
    # Copies script and stores it in path
    # https://stackoverflow.com/questions/23321100/best-way-to-have-a-python-script-copy-itself/49210778
    copied_script_name = time.strftime("%Y-%m-%d_%H%M") + '_' + os.path.basename(__file__)
    copied_path = (path / copied_script_name).absolute()
    shutil.copyfile(__file__, copied_path)
    print(f"Saved a copy of script to {copied_path}")


def create_electrode_plot(rec_name, results_path, matlab_dict):
    ######################################################
    ############  ELECTRODE PLOT PARAMETES  ##############
    ######################################################
    # Unit (circle) areas scale linearly with num_spikes

    # Size of unit = num_spikes * ELECTRODE_SPIKE_SIZE_SCALE
    ELECTRODE_SPIKE_SIZE_SCALE = 1
    # Alpha value of units (0 = fully transparent, 1 = fully opaque)
    ELECTRODE_SPIKE_ALPHA = 0.7
    # Color of units (in hexadecimal)
    ELECTRODE_SPIKE_COLOR = "#000000"
    # For each value (n) in ELECTRODE_SPIKE_LEGEND_SCALES, a circle will appear on a legend
    # that indicates the size of the unit if the unit had n spikes
    ELECTRODE_SPIKE_LEGEND_SCALES = [100, 500, 1000]
    # Size of dot when plotting electrode locations
    ELECTRODE_DOT_SIZE = 1
    # Color of dot when plotting electrode locations
    ELECTRODE_DOT_COLOR = "#000000"
    # Buffer for setting limits of x-axs
    # The axis will go from -ELECTRODE_X_MAX_BUFFER to maximum_x_value_of_data + ELECTRODE_X_MAX_BUFFER
    ELECTRODE_X_MAX_BUFFER = 100
    # Buffer for setting limits of y-axs
    # The axis will go from -ELECTRODE_Y_MAX_BUFFER to maximum_y_value_of_data + ELECTRODE_Y_MAX_BUFFER
    ELECTRODE_Y_MAX_BUFFER = 100
    # Label for x-axis
    ELECTRODE_X_LABEL = "x (μm)"
    # Label for y-axis
    ELECTRODE_Y_LABEL = "y (μm)"

    stopwatch = Stopwatch("CREATING ELECTRODE PLOT")
    # Get plotting data
    loc_sizes = OrderedDict()
    loc_x = []
    loc_y = []
    for unit_dict in matlab_dict["units"]:
        loc = (unit_dict["x_max"], unit_dict["y_max"])
        size_increase = ELECTRODE_SPIKE_SIZE_SCALE * len(unit_dict["spike_train"])
        if loc in loc_sizes:
            loc_sizes[loc] += size_increase
        else:
            loc_sizes[loc] = size_increase
            loc_x.append(loc[0])
            loc_y.append(loc[1])

    # Create plot
    fig, ((a00, a01), (a10, a11)) = plt.subplots(2, 2)  # type:figure.Figure, ((axes.Axes, axes.Axes), (axes.Axes, axes.Axes))
    fig.delaxes(a10)
    fig.suptitle(f"Electrode Plots of {rec_name}")

    # Plot recording MEA
    channel_locations = matlab_dict["locations"]
    x_max = max(channel_locations[:, 0])
    y_max = max(channel_locations[:, 1])
    n_channels = channel_locations.shape[0]
    a00.scatter(channel_locations[:, 0], channel_locations[:, 1], [ELECTRODE_DOT_SIZE]*n_channels,
                c=ELECTRODE_DOT_COLOR)

    a00.set_title(f"MEA - {n_channels} channels")
    a00.set_xlabel(ELECTRODE_X_LABEL)
    a00.set_ylabel(ELECTRODE_Y_LABEL)
    a00.set_aspect("equal")
    a00.set_xlim(-ELECTRODE_X_MAX_BUFFER, x_max+ELECTRODE_X_MAX_BUFFER)
    a00.set_ylim(-ELECTRODE_Y_MAX_BUFFER, y_max+ELECTRODE_Y_MAX_BUFFER)

    # Plot units
    a01.scatter(loc_x, loc_y, loc_sizes, c=ELECTRODE_SPIKE_COLOR, alpha=ELECTRODE_SPIKE_ALPHA)
    a01.set_title(f"Spike Locations")
    a01.set_xlabel(ELECTRODE_X_LABEL)
    a01.set_ylabel(ELECTRODE_Y_LABEL)
    a01.set_aspect("equal")
    a01.set_xlim(*a00.get_xlim())
    a01.set_ylim(*a00.get_ylim())

    # Plot size legend
    loc_off_screen = (-1, -1)
    a11.set_xlim(0, 0.1)
    a11.set_ylim(0, 0.1)
    a11.axis("off")
    # Area scales linearly
    for size in ELECTRODE_SPIKE_LEGEND_SCALES:
        a11.scatter(*loc_off_screen, size * ELECTRODE_SPIKE_SIZE_SCALE,
                    color=ELECTRODE_SPIKE_COLOR, alpha=ELECTRODE_SPIKE_ALPHA)
    a11.legend(loc="center", labelspacing=2, borderpad=1)
    fig.savefig(results_path / (rec_name+"_electrode_plots.png"))
    fig.clf()
    stopwatch.log_time("Done.")


def create_figures(save_folder, rec_names, waveform_extractors):
    save_folder = Path(save_folder) / "figures"

    total_stopwatch = Stopwatch("CREATING FIGURES")

    if not AUTO_CREATE_FIGURES:
        print("'AUTO_CREATE_FIGURES' is set to False, so figures will not be created")
        return

    print(f"Plotting data from the following recordings: {', '.join(rec_names)}")

    bar_data_n_total = []
    bar_data_n_curated = []

    scatter_data = [
        [],  # n_spikes
        [],  # std
        [],  # color
    ]
    scatter_n_spikes_max = 0
    scatter_std_norm_max = 0
    scatter_legend_colors = []

    all_templates_unit_ids = np.array([])
    all_templates_is_curated = np.array([])
    all_templates_templates_all = None

    extract_data_stopwatch = Stopwatch()
    for (rec_name, w_e) in zip(rec_names, waveform_extractors):
        print_stage("EXTRACTING PLOTTING DATA")
        print(f"from {rec_name}")

        curation_history = w_e.get_curation_history()

        key_curation_final = curation_history["curations"][-1]

        if AUTO_CURATE_WAVEFORMS:
            stopwatch = Stopwatch("\nGetting data for curation bar plot", use_print_stage=False)
            n_curated = len(curation_history["curated"][key_curation_final])
            n_total = n_curated + len(curation_history["failed"][key_curation_final])
            bar_data_n_total.append(n_total)
            bar_data_n_curated.append(n_curated)
            stopwatch.log_time("Done.")
        else:
            print("\nSkipping creating curation bar plot because 'AUTO_CURATE_WAVEFORMS' is set to False")

        if AUTO_CURATE_WAVEFORMS and SPIKES_PER_UNIT_MIN is not None and STD_NORM_MAX is not None:
            stopwatch = Stopwatch("\nGetting data for std scatter plot", use_print_stage=False)
            n_spikes = curation_history["metrics"]["spikes_min"]
            stds = curation_history["metrics"]["std_norm_max"]
            color = SCATTER_RECORDING_COLORS.pop(0)

            if SCATTER_STD_MAX_UNITS_PER_RECORDING is not None:
                n_units = len(n_spikes)
                if SCATTER_STD_MAX_UNITS_PER_RECORDING < n_units:
                    random_sample = np.random.choice(n_units, replace=False, size=SCATTER_STD_MAX_UNITS_PER_RECORDING)
                    n_spikes = np.asarray(n_spikes)[random_sample].tolist()
                    stds = np.asarray(stds)[random_sample].tolist()

            scatter_n_spikes_max = max(scatter_n_spikes_max, *n_spikes)
            scatter_std_norm_max = max(scatter_std_norm_max, *stds)
            scatter_data[0].extend(n_spikes)
            scatter_data[1].extend(stds)
            scatter_data[2].extend(len(n_spikes) * [color])
            scatter_legend_colors.append((rec_name, color))
            stopwatch.log_time("Done.")
        else:
            print("\nSkipping creating std scatter plot because")
            print("'AUTO_CURATE_WAVEFORMS' is False OR 'SPIKES_PER_UNIT_MIN' is None OR 'STD_NORM_MAX' is None")

        stopwatch = Stopwatch("\nGetting data for plotting all templates", use_print_stage=False)
        unit_ids_curated = curation_history["curated"][key_curation_final]
        unit_ids_failed = curation_history["failed"][key_curation_final]
        all_unit_ids = unit_ids_curated + unit_ids_failed

        templates = w_e.get_computed_template(all_unit_ids, mode="average")[:, :, w_e.chans_max_all[all_unit_ids]]

        all_templates_unit_ids = np.concatenate((all_templates_unit_ids, unit_ids_curated, unit_ids_failed))
        all_templates_is_curated = np.concatenate((all_templates_is_curated, [1]*len(unit_ids_curated), [0]*len(unit_ids_failed)))
        if all_templates_templates_all is None:
            all_templates_templates_all = templates
        else:
            all_templates_templates_all = np.concatenate((all_templates_templates_all, templates), axis=0)

        stopwatch.log_time("Done.")
    extract_data_stopwatch.log_time("\nDone extracting plotting data from all recordings. Total")

    print_stage("PLOTTING DATA")
    if AUTO_CURATE_WAVEFORMS:
        stopwatch = Stopwatch("Plotting bar plot", use_print_stage=False)
        df = pd.DataFrame({"Total Units": bar_data_n_total,
                           "Selected Units": bar_data_n_curated},
                          index=rec_names)
        df.plot.bar(rot=0)  # rot = rotation of bar labels. i.e. rot=0 means labels have no rotation
        plt.xlabel(BAR_X_LABEL)
        plt.ylabel(BAR_Y_LABEL)
        fig_path = save_folder / "curation_bar_plot.png"
        plt.savefig(fig_path)
        plt.clf()
        print(f"Saved curation bar plot figure to {fig_path}")
        stopwatch.log_time("Done.")

    if AUTO_CURATE_WAVEFORMS and SPIKES_PER_UNIT_MIN is not None and STD_NORM_MAX is not None:
        stopwatch = Stopwatch("\nPlotting std scatter plot", use_print_stage=False)
        x_max = scatter_n_spikes_max + SCATTER_X_MAX_BUFFER
        y_max = scatter_std_norm_max + SCATTER_Y_MAX_BUFFER

        # Plot data
        indices = np.arange(len(scatter_data[0]))
        np.random.shuffle(indices)
        n_spikes = np.asarray(scatter_data[0])[indices]
        stds = np.asarray(scatter_data[1])[indices]
        colors = np.asarray(scatter_data[2])[indices]
        plt.scatter(n_spikes, stds, colors)

        threshold_lines_kwargs = {"linestyle": "dotted", "linewidth": 1, "c": "#000000"}
        # Threshold for minimum spikes per unit line
        plt.axvline(SPIKES_PER_UNIT_MIN, **threshold_lines_kwargs)
        plt.text(SPIKES_PER_UNIT_MIN, y_max, SPIKES_PER_UNIT_MIN, horizontalalignment="center")
        # Threshold for maximum norm std line
        plt.axhline(STD_NORM_MAX, **threshold_lines_kwargs)
        plt.text(x_max, STD_NORM_MAX, STD_NORM_MAX, verticalalignment="center")

        # Set labels
        plt.xlim(0, x_max)
        plt.ylim(0, y_max)
        plt.xlabel(SCATTER_X_LABEL)
        plt.ylabel(SCATTER_Y_LABEL)

        # Create legend
        off_screen = (-1e4, -1e4)
        for name, c in scatter_legend_colors:
            plt.scatter(*off_screen, c=c)
        plt.legend()

        fig_path = save_folder / "std_scatter_plot.png"
        plt.savefig(fig_path)
        plt.clf()
        print(f"Saved std scatter plot figure to {fig_path}")
        stopwatch.log_time("Done.")

    stopwatch = Stopwatch("\nPlotting templates", use_print_stage=False)
    print("Sorting amplitudes")
    amplitudes = all_templates_templates_all[:, waveform_extractors[0].nbefore]
    amplitudes.

    fig_path = save_folder / "all_templates_plot.png"
    plt.savefig(fig_path)
    plt.clf()
    print(f"Saved all templates figure to {fig_path}")
    stopwatch.log_time("Done.")

    print_stage("DONE CREATING ALL FIGURES")
    total_stopwatch.log_time("Total")


def main():
    figures_rec_names = []
    figures_waveform_extractors = []

    for (rec_path, inter_path, results_path) in zip(RECORDING_FILES, INTERMEDIATE_FOLDERS, RESULTS_FOLDERS):
        stopwatch = Stopwatch()
        result = process_recording(rec_path, inter_path)
        if type(result) == str:
            print(result)
            continue

        rec_name = str(rec_path).split(r"/")[-1].split(".")[0]

        convert_to_matlab(result, rec_name, results_path)

        print_stage(f"DONE WITH RECORDING")
        print(f"Recording: {rec_path}")
        stopwatch.log_time("Total")
        print("\n\n")

        figures_rec_names.append(rec_name)
        figures_waveform_extractors.append(result)

    create_figures(RESULTS_FOLDERS[-1], figures_rec_names, figures_waveform_extractors)


if __name__ == "__main__":
    main()
# endregion
