from typing import NamedTuple
from utils import array_from_csv


# (field_name, field_type_converter)
detobj_field_type = [("frame_index", int),
                     ("det_id", int),
                     ("label", str),
                     ("score", float),
                     ("box2d", array_from_csv),
                     ("image_size", array_from_csv)]


class DetObject(NamedTuple("DetObject", detobj_field_type)):

    """
    Class for holding detections from convolutional neural networks.
    """

    def get_bbox_center(self):
        x1, y1, x2, y2 = self.box2d
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return cx, cy
