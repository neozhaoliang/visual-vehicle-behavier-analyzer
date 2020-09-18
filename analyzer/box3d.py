"""
3D bounding box estimation of a detected object.
"""
import numpy as np
from typing import NamedTuple
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from utils import create_mask_from_pixels, mask_rect_diff


# rough size of the objects in real world (in meters)
prior_knowledge = {
    "car": (4.8, 2, 1.6),
    "person": (1, 1, 1.8),
    "bicycle": (2, 0.5, 1),
    "truck": (8, 3, 2),
    "motorbike": (2.5, 1, 1.6),
    "bus": (10, 3, 2)
}

# (field_names, field_types)
box3d_field_types = [
    ("yaw", float),
    ("x", float),
    ("y", float),
    ("z", float),
    ("l", float),
    ("w", float),
    ("h", float),
    ("frame_index", int),
    ("det_id", int),
]


class Box3D(NamedTuple("Box3D", box3d_field_types)):

    """
    yaw : euler angle about z-axis of the object, in radians.
    x, y, z : bottom center of the 3d box in world frame.
    l, w, h : lenght, width, height of the 3d box in meters.

    This class has a similar interface with namedtuple so one can
    read/write it from csv in the same way with namedtuple objects.
    """

    @property
    def yaw(self):
        return self[0]

    @property
    def xyz(self):
        return self[1: 4]

    @property
    def lwh(self):
        return self[4: 7]

    @property
    def corners(self):
        """
        Get the eight corners of the 3d box.
        """
        l, w, h = self.lwh
        rot = R.from_euler("z", self.yaw, degrees=False)
        points = []
        for dx in [-l/2, l/2]:
            for dy in [-w/2, w/2]:
                for dz in [0, h]:
                    p = [dx, dy, dz]
                    c = np.add(self.xyz, rot.apply(p))
                    points.append(c)
        return points

    def get_mask_on_image(self, camera, image_size):
        """
        Get the mask of the convex hull formed by the 8
        pixels corresponding to the box corners.
        """
        pixels = camera.world_to_image(self.corners)
        return create_mask_from_pixels(pixels, *image_size)

    def get_bbox_on_image(self, camera):
        """
        Get a 2d bounding rect on image formed by the 8 pixels
        corresponding to the box corners.
        """
        pixels = camera.world_to_image(self.corners)
        x, y, w, h = cv2.boundingRect(pixels)
        return np.array([x, y, x + w, y + h])

    def __str__(self):
        desc = (
            "Box3D object:\n" +
            "frame index: {}".format(self.frame_index) +
            "detection id: {}".format(self.det_id) +
            "yaw: {}\n".format(np.degrees(self.yaw)) +
            "world coordinates: ({}, {}, {})\n".format(*self.xyz) +
            "3d bounding box: ({}, {}, {})".format(*self.lwh)
        )
        return desc


def get_box3d_from_detobj(det_obj, camera):
    """
    Get an optimized estimation of the 3d box of a detected object.

    det_obj: an instance of the `DetObject` class.
    camera : an instance of the `PinholeCamera` class.
    """
    print("=" * 48)
    print("[INFO] computing 3d bounding box of {} at {}".format(
        det_obj.label, det_obj.box2d))

    def func_cost(params_opt, params_fix, camera, image_size, yolo_bbox):
        """
        Compute the cost of the estimated 3d box by projecting it back to
        image pixels and compare its difference with the bbox detected by
        YOLOv3.
        """
        a = 0.2
        box3d_obj = Box3D(*params_opt, *params_fix, det_obj.frame_index, det_obj.det_id)
        box_mask = box3d_obj.get_mask_on_image(camera, image_size)
        overlap, diff1, diff2 = mask_rect_diff(box_mask, yolo_bbox)
        return a * diff1  + (1 - a) * diff2


    # project the center of the 2d box to ground as initial position
    x1, y1, x2, y2 = det_obj.box2d
    pt = ((x1 + x2) // 2, (y1 + y2) // 2)
    p0, = camera.image_to_world([pt], -prior_knowledge[det_obj.label][2]/2)
    x, y, z = p0
    l, w, h = prior_knowledge[det_obj.label]

    params_min = None
    for yaw in np.linspace(0, np.pi, 6):
        params_opt = [yaw, x, y]
        params_fix = [z, l, w, h]
        param = minimize(
            func_cost, params_opt, method="Powell",
            args=(params_fix, camera, det_obj.image_size, det_obj.box2d),
            options={"maxfev": 2000, "disp": True}
        )

        if params_min is None or params_min.fun > param.fun:
            params_min = param

    return Box3D(*params_min.x[:3], *params_fix, det_obj.frame_index, det_obj.det_id)
