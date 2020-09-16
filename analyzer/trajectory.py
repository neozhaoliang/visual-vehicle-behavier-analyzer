import numpy as np
from .kalman import BicycleModel, CVModel


class TrajectoryProcessor(object):

    """
    Use a calibrated camera to convert pixel locations of a track to its
    world coordinates.
    """

    def __init__(self, track, camera):
        """
        track: an instance of `Track` class.
        camera: an instance of `PinholeCamera` class.
        """
        self.track = track
        self.camera = camera
        self.trajectory = None
        self.pixel_history = []

    def get_pixel_by_frame(self, frame_index):
        """
        Return the center of the bounding box of its track at frame=`frame_index`.
        """
        detobj = self.track.get_history_detection_by_frame(frame_index)
        if detobj is not None:
            return detobj.get_bbox_center()
        return None

    def get_world_position_by_frame(self, frame_index):
        pixel = self.get_pixel_by_frame(frame_index)
        if pixel is not None:
            return self.camera.image_to_world([pixel])[0]
        return None

    def process_trajectory(self, list_times_ms):
        # get the initial and last frame index
        i0 = self.track.init_frame_index
        i1 = self.track.last_frame_index

        # initial timestamp
        t0 = list_times_ms[i0]
        # initial world location
        p0 = self.get_world_position_by_frame(i0)

        # run constant velocity filter model
        ekf_cv = CVModel(x0=[p0[0], p0[1], 0, 0], P0=np.diag([0.1, 0.1, 10, 10]),
                         Q=np.diag([0, 0, 0.1, 0.1]), R=np.diag([10, 10]), timestamp=t0)
        ekf_cv.start()

        for frame_index in range(i0 + 1, i1 + 1):
            current_time_ms = list_times_ms[frame_index]
            pixel = self.get_pixel_by_frame(frame_index)
            ekf_cv.predict_and_update(current_time_ms, pixel, self.camera)

        # use the smoothed state as the initial estimation for the bicycle model
        ekf_cv.rts_interval_smoothing()
        _, x0_cv_smoothed, _ = ekf_cv.smooth_history[0]
        x0 = ekf_cv.cv2bicycle(x0_cv_smoothed)

        ekf_bm = BicycleModel(x0=x0, P0=np.diag([1, 1, 2, 1, 0.1]),
                              Q=np.diag([0, 0, 1, 0, 0.01]), R=np.diag([10, 10]), timestamp=t0)
        ekf_bm.start()

        for frame_index in range(i0 + 1, i1 + 1):
            current_time_ms = list_times_ms[frame_index]
            pixel = self.get_pixel_by_frame(frame_index)
            ekf_bm.predict_and_update(current_time_ms, pixel, self.camera)

        ekf_bm.rts_interval_smoothing()
        self.trajectory = ekf_bm.smooth_history

        # compute the corresponding pixel for each smoothed state
        for _, x_smooth, _ in self.trajectory:
            x, y = x_smooth[:2, 0]
            pix, = self.camera.world_to_image([x, y, 0.6])
            self.pixel_history.append(pix)

    def _asDict(self):
        x = np.array([x_smooth[:4, 0] for _, x_smooth, _ in self.trajectory]).T
        di = {
            "start_frame": self.track.init_frame_index,
            "end_frame": self.track.last_frame_index,
            "label": self.track.agent_type,
            "world_x": x[0],
            "world_y": x[1],
            "velocity": x[2],
            "yaw": x[3]
        }
        return di
