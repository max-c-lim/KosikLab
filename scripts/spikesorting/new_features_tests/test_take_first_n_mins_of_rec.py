from spikeinterface.extractors import MaxwellRecordingExtractor

FIRST_N_MINS = 5

rec = MaxwellRecordingExtractor("/home/maxlim/kosik_server/data/MEAprojects/organoid/220705/16460/Network/000439/data.raw.h5")
print(f"Total duration (in s): {rec.get_total_duration()}")
print(f"Total duration (in m): {rec.get_total_duration() / 60}")

end_frame = FIRST_N_MINS * 60 * rec.get_sampling_frequency()
rec_cut = rec.frame_slice(start_frame=0, end_frame=end_frame)
print(f"Cut end frame: {end_frame}")
print(f"Cut duration (in s): {rec_cut.get_total_duration()}")
print(f"Cut duration (in m): {rec_cut.get_total_duration() / 60}")

