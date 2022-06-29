# Testing forcing BinaryRecordingExtractor.write_binary_recording to return scaled traces

import scipy.signal
from spikeinterface.toolkit.preprocessing.filter import FilterRecordingSegment as _FilterRecordingSegment


class FilterRecordingSegment(_FilterRecordingSegment):
    def __init__(self, _filter_recording_segment, rec_filtered):
        super().__init__(_filter_recording_segment.parent_recording_segment, _filter_recording_segment.coeff,
                         _filter_recording_segment.filter_mode, _filter_recording_segment.margin,
                         _filter_recording_segment.dtype)

        if rec_filtered.has_scaled_traces():
            self.gain_to_uV = rec_filtered.get_property('gain_to_uV').astype('float32')
            self.offset_to_uV = rec_filtered.get_property('offset_to_uV').astype('float32')
        else:
            self.gain_to_uV = 1
            self.offset_to_uV = 0

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces_chunk, left_margin, right_margin = FilterRecordingSegment.get_chunk_with_margin(self.parent_recording_segment,
                                                                        start_frame, end_frame, channel_indices,
                                                                        self.margin)
        traces_chunk = traces_chunk * self.gain_to_uV + self.offset_to_uV

        traces_dtype = traces_chunk.dtype
        # if uint --> force int
        if traces_dtype.kind == "u":
            traces_chunk = traces_chunk.astype("float32")

        if self.filter_mode == 'sos':
            filtered_traces = scipy.signal.sosfiltfilt(self.coeff, traces_chunk, axis=0)
        elif self.filter_mode == 'ba':
            b, a = self.coeff
            filtered_traces = scipy.signal.filtfilt(b, a, traces_chunk, axis=0)

        if right_margin > 0:
            filtered_traces = filtered_traces[left_margin:-right_margin, :]
        else:
            filtered_traces = filtered_traces[left_margin:, :]
        return filtered_traces.astype(self.dtype)
    @staticmethod
    def get_chunk_with_margin(rec_segment, start_frame, end_frame,
                              channel_indices, margin, add_zeros=False):
        """
        Helper to get chunk with margin
        """
        length = rec_segment.get_num_samples()

        if channel_indices is None:
            channel_indices = slice(None)

        if not add_zeros:
            if start_frame is None:
                left_margin = 0
                start_frame = 0
            elif start_frame < margin:
                left_margin = start_frame
            else:
                left_margin = margin

            if end_frame is None:
                right_margin = 0
                end_frame = length
            elif end_frame > (length - margin):
                right_margin = length - end_frame
            else:
                right_margin = margin
            traces_chunk = rec_segment.get_traces(start_frame - left_margin, end_frame + right_margin, channel_indices)

        else:
            # add_zeros=True
            assert start_frame is not None
            assert end_frame is not None
            chunk_size = end_frame - start_frame
            full_size = chunk_size + 2 * margin

            if start_frame < margin:
                start_frame2 = 0
                left_pad = margin - start_frame
            else:
                start_frame2 = start_frame - margin
                left_pad = 0

            if end_frame > (length - margin):
                end_frame2 = length
                right_pad = end_frame + margin - length
            else:
                end_frame2 = end_frame + margin
                right_pad = 0

            traces_chunk = rec_segment.get_traces(start_frame2, end_frame2, channel_indices)

            if left_pad > 0 or right_pad > 0:
                traces_chunk2 = np.zeros((full_size, traces_chunk.shape[1]), dtype=traces_chunk.dtype)
                i0 = left_pad
                i1 = left_pad + traces_chunk.shape[0]
                traces_chunk2[i0: i1, :] = traces_chunk
                left_margin = margin
                if end_frame < (length + margin):
                    right_margin = margin
                else:
                    right_margin = end_frame + margin - length
                traces_chunk = traces_chunk2
            else:
                left_margin = margin
                right_margin = margin

        return traces_chunk, left_margin, right_margin

import numpy as np
from typing import Union, List
from spikeinterface.extractors.neoextractors.neobaseextractor import NeoRecordingSegment as _NeoRecordingSegment
class NeoRecordingSegment(_NeoRecordingSegment):
    """
    for i, rec_seg in enumerate(rec._recording_segments):
        rec._recording_segments[i] = NeoRecordingSegment(rec_seg, rec)

    """

    def __init__(self, _neo_recording_segmment, rec):
        super().__init__(_neo_recording_segmment.neo_reader, _neo_recording_segmment.segment_index,
                         _neo_recording_segmment.stream_index)
        self._neo_recording_segment = _neo_recording_segmment
        if rec.has_scaled_traces():
            self.gains = rec.get_property('gain_to_uV').astype('float32')
            self.offsets = rec.get_property('offset_to_uV').astype('float32')
        else:
            self.gains = 1
            self.offsets = 0

    def get_traces(self,
                   start_frame: Union[int, None] = None,
                   end_frame: Union[int, None] = None,
                   channel_indices: Union[List, None] = None,
                   ) -> np.ndarray:
        traces = self._neo_recording_segment.get_traces(start_frame=start_frame, end_frame=end_frame, channel_indices=channel_indices)
        return traces * self.gains + self.offsets