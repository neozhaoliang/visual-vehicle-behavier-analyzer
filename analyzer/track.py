"""
Associate detections in a stream of frames and merge different tracks.

Example:
>>> asso = TrackDetAssociation()
>>> for image in image_list:
...     asso.update(detections_from_this_image)
>>> merged_tracks = CVTrackMerger.run_merge_tracks(asso.tracks, zone)
"""
import numpy as np
import cv2
from scipy.stats import mode
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
import tqdm
from utils import bbox_to_xywh, xywh_to_bbox, iou_rect, iou_det_object


class Track2D(object):

    """
    Class holding the tracking information of an object in a stream
    of frames (possibly with missing detections).
    """
    __slots__ = ("track_id", "agent_type", "init_frame_index",
                 "last_frame_index", "det_history")

    def __init__(self, track_id, agent_type):
        """
        track_id : integer, an unique id for this tracked object.
        agent_type : str, label of this tracked object, e,g. car.

        attributes:

        init_frame_index : the frame index that this track starts.
        last_frame_index : the frame index that this track ends.

        Both these two indices may vary if new detections are added.

        det_history : list of history detections, each item in this
            list is either `None` (if it's not detected in the
            corresponding frame) or an instance of the `DetObject`.
        """
        self.track_id = track_id
        self.agent_type = agent_type
        self.init_frame_index = None
        self.last_frame_index = None
        self.det_history = []

    def __len__(self):
        return len(self.det_history)

    def get_true_detections(self, reverse=False):
        """
        Traverse the detection history list and skip `None` detections.
        """
        if reverse:
            iterable = reversed(self.det_history)
        else:
            iterable = self.det_history

        for detobj in iterable:
            if detobj is not None:
                yield detobj

    def update_agent_type(self):
        """
        Set the label with highest occurrence in detection history.
        """
        if len(self) > 0:
            labels = [detobj.label for detobj in self.get_true_detections()]
            self.agent_type =  mode(labels)[0][0]
        return self.agent_type

    def update(self, detobj):
        """
        Insert a new `DetObject` into the detection history.
        """
        if detobj is None:
            return

        frame_index = detobj.frame_index

        # if this track has not been binded to any detection yet
        if self.init_frame_index is None:
            self.init_frame_index = self.last_frame_index = frame_index
            self.det_history.append(detobj)

        # replace a known detection (possibly `None`) with this new one
        elif self.init_frame_index <= frame_index <= self.last_frame_index:
            index = frame_index - self.init_frame_index
            self.det_history[index] = detobj

        # extend detection history to the future
        elif frame_index > self.last_frame_index:
            padding = [None] * (frame_index - self.last_frame_index - 1)
            self.det_history += padding
            self.det_history.append(detobj)
            self.last_frame_index = frame_index

        # extend detection history to the past
        else:
            padding = [None] * (self.init_frame_index - frame_index - 1)
            self.det_history = padding + self.det_history
            self.det_history.insert(0, detobj)
            self.init_frame_index = frame_index

    def get_history_index_by_frame(self, frame_index):
        """
        A convertion between frame index and detection history index.
        """
        if len(self) == 0:
            raise ValueError("Empty track: {}".format(self))

        index = frame_index - self.init_frame_index
        if 0 <= index < len(self):
            return index
        return None

    def get_history_detection_by_frame(self, frame_index):
        """
        Return the `DetObject` instance in the detection history list
        which was detected in the frame of index=`frame_index`.
        """
        index = self.get_history_index_by_frame(frame_index)
        if index is not None:
            return self.det_history[index]
        return None

    def merge_track(self, another_track):
        """
        Merge this track with another one.

        another_track : another instance of `Track2D`.
        """
        for detobj in another_track.get_true_detections():
            self.update(detobj)

    def __str__(self):
        indices = [detobj.frame_index for detobj in self.get_true_detections()]
        return ("Track2D object:\n" +
                "agent type: {}\n".format(self.update_agent_type()) +
                "occurred frame indices list: {}".format(indices))

    def roll_back(self, num_past_frames):
        """
        Return the detection of this track at `num_past_frames` ahead of its last detection.
        Note `None` detections are not counted.
        """
        if len(self) == 0:
            raise ValueError("Empty track: {}".format(self))

        for k, detobj in enumerate(self.get_true_detections(reverse=True)):
            if k == num_past_frames:
                break
        return detobj


def associate_tracks_and_detections(track_list, det_list, iou_threshold):
    """
    Associate two lists of detections from two different frames,
    all items in both lists must be instances of `DetObject`.
    `track_list` is a list of tracks we are currently tracking,
    `det_list` is a list of detections in the current frame.
    """
    matches = []
    num_trks = len(track_list)
    num_dets = len(det_list)
    if num_dets > 0 and num_trks > 0:
        cost_mat = np.zeros((num_trks, num_dets))

        # assign the entries of the cost matrix by the iou
        # of the pairs of `DetObject`s.
        for track_ind, track in enumerate(track_list):
            for det_ind, det in enumerate(det_list):
                if (track is not None and det is not None):
                    cost_mat[track_ind, det_ind] = iou_det_object(track, det)

        # check the matched pairs have an iou above the threshold
        matched_indices = linear_sum_assignment(-cost_mat)
        for i, j in zip(*matched_indices):
            if cost_mat[i, j] > iou_threshold:
                # here we put the index of a track and its corresponding
                # `DetObject` instance into a pair, it looks weird but
                # will proved to be a very convenient way for looping and
                # matching multiple times over a fixed `det_list`.
                matches.append((i, det_list[j]))
    # return the indices of the matched pairs
    return matches


class TrackDetAssociation(object):

    """
    Class for associating detections from the current frame with
    existing tracks.
    """

    def __init__(self, iou_threshold=0.3, max_past_frames=30):
        """
        iou_threshold : threshold used in iou matching.
        max_past_frames : the max number of frames looked into the past
            when trying to associate current detections with existing tracks.
        """
        self.tracks = []  # all tracks (including dead tracks)
        self.active_tracks = []  # only live tracks
        self.iou_threshold = iou_threshold
        self.max_past_frames = max_past_frames
        self.track_index = 0
        self.frame_index = 0

    def prepare_for_match(self, frame_index):
        """
        For each `Track2D` instance in the active tracks list, select the
        detection at frame of index=`frame_index` and return the list
        formed by these detections.
        """
        result_tracks = []
        for track in self.active_tracks:
            item = None
            # filter out those tracks that has already beeen matched with detections,
            # such tracks must have their last_frame_index equals to current frame index.
            if track.last_frame_index != self.frame_index:
                item = track.get_history_detection_by_frame(frame_index)
            result_tracks.append(item)
        return result_tracks

    def update(self, detections_list):
        """
        Update the tracks by a list of detections from a new frame.
        """
        num_tracks = len(self.active_tracks)
        num_dets = len(detections_list)
        if num_tracks > 0 and num_dets > 0:
            # loop over a fixed number of past frames and try to
            # match the tracks and detections.
            for past_frame_index in range(
                self.frame_index,
                max(self.frame_index - self.max_past_frames, 0) - 1,
                -1):
                track_list = self.prepare_for_match(past_frame_index)
                pairs = associate_tracks_and_detections(
                    track_list,
                    detections_list,
                    self.iou_threshold
                )
                # note each pair consists of an integer and an `DetObj`
                for track_ind, det_obj in pairs:
                    # update the track by its matching detection
                    self.active_tracks[track_ind].update(det_obj)
                    # remove the matched detection
                    detections_list.remove(det_obj)

        # create a new track for each unmatched detection
        for unmatched_det in detections_list:
            new_track = Track2D(self.track_index, unmatched_det.label)
            new_track.update(unmatched_det)
            self.active_tracks.append(new_track)
            self.tracks.append(new_track)
            self.track_index += 1

        # delete "dead" tracks which are tracks that has not been updated for
        # a while
        for track in self.active_tracks:
            if self.frame_index - track.last_frame_index > self.max_past_frames:
                self.active_tracks.remove(track)

        self.frame_index += 1


class CVTrackMerger(object):

    """
    Use OpenCV visual trackers to merge the tracks.
    """

    PREVIOUS_HORIZON = 30
    HORIZON_INTERVAL = 60
    OVERLAP_MIN = 0.3
    OMIT_SHORT_THRESOLD = 30

    def __init__(self, video_capture):
        """
        video_capture: an instance of `cv2.VideoCapture` class.
        """
        self.capture = video_capture
        self.nframes = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def create_visual_tracker(self, track):
        """
        Create a visual tracker from a given `Track2D` instance.
        The initial frame is choosen at `PREVIOUS_HORIZON` frames
        ahead of the last frame of `track`.
        """
        # roll back `PREVIOUS_HORIZON` frames and find the detection there
        detobj = track.roll_back(self.PREVIOUS_HORIZON)
        # create a visual tracker using opencv's CSRT tracker
        cv_tracker = cv2.TrackerCSRT_create()
        # initialize this tracker using `detobj`
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, detobj.frame_index)
        _, frame = self.capture.read()
        bbox = detobj.box2d
        cv_tracker.init(frame, bbox_to_xywh(bbox))
        return detobj, cv_tracker

    def run_merge_tracks(self, track_list, zone=None):
        merged_pairs = []
        total = len(track_list)
        for track in track_list:
            print("[INFO] finding candidates for track {}/{}".format(track.track_id, total))
            detobj, cv_tracker = self.create_visual_tracker(track)
            candidates = []
            start = detobj.frame_index
            end = min(start + self.HORIZON_INTERVAL, self.nframes)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            for frame_index in tqdm.trange(start, end):
                _, frame = self.capture.read()
                success, roi = cv_tracker.update(frame)
                if success:
                    bbox = xywh_to_bbox(roi)
                    cand_track = self.choose_candidate(track, bbox, frame_index, track_list, candidates)
                    if cand_track is not None:
                        candidates.append(cand_track.track_id)
                else:
                    break

            if len(candidates) > 0:
                opt_cand = mode(candidates)[0][0]
                merged_pairs.append((track.track_id, opt_cand))

        return self.collect(track_list, merged_pairs, zone)

    def valid_candidate(self, frame_index, track, cand_track):
        """
        Check if we should match two tracks at a given frame.
        """
        # they must have different id's.
        cond1 = (track.track_id != cand_track.track_id)
        # the life span of `cand_track` must cover the frame of index=`frame_index
        cond2 = (cand_track.init_frame_index <= frame_index <= cand_track.last_frame_index)
        # they cannot co-exist before `track`'s visual tracker is created.
        cond3 = (cand_track.init_frame_index >= track.last_frame_index - self.PREVIOUS_HORIZON)
        # `frame_index` should not be too far in the second track's history
        cond4 = frame_index <= cand_track.init_frame_index + 30
        return all([cond1, cond2, cond3, cond4])

    def choose_candidate(self, track0, bbox, frame_index, track_list, candidate_list):
        for track in track_list:
            if track.track_id in candidate_list or self.valid_candidate(frame_index, track0, track):
                detobj = track.get_history_detection_by_frame(frame_index)
                if detobj is not None:
                    iou = iou_rect(bbox, detobj.box2d)
                    if iou > self.OVERLAP_MIN:
                        return track
        return None

    def collect(self, track_list, merged_pairs, zone):
        """
        Merge tracks by finding the connected components of the graph
        defined by the edges in `merged_pairs`.
        """
        final_tracks = []

        # adjacent matrix of the graph, the vertices are the tracks
        # and the edges are pairs given in `merged_pairs`.
        N = len(track_list)
        graph = np.zeros((N, N), dtype=int)
        for i, j in merged_pairs:
            graph[i, j] = graph[j, i] = 1

        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

        # merge each component into its first track
        for k in range(n_components):
            group = [trk for i, trk in enumerate(track_list) if labels[i] == k]
            trk0 = group[0]
            for trk in group[1:]:
                trk0.merge_track(trk)

            # discard this track if it's too short
            if len(trk0) >= self.OMIT_SHORT_THRESOLD:
                if zone is None or self.is_interested_track(trk0, zone):
                    final_tracks.append(trk0)

        # reset the id of these final tracks and update their labels
        for k, trk in enumerate(final_tracks):
            trk.track_id = k
            trk.update_agent_type()

        return final_tracks

    def is_interested_track(self, track, zone):
        for detobj in track.det_history:
            if detobj is not None:
                center = detobj.get_bbox_center()
                if zone.in_zone(center):
                    return True
        return False
