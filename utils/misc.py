import subprocess
import cv2
import numpy as np
import csv
from matplotlib.cm import cmap_d


def create_mask_from_pixels(pixels, width, height):
    """
    Create mask from the convex hull of a list of pixels.
    """
    pixels = np.int32(pixels).reshape(-1, 2)
    hull = cv2.convexHull(pixels)
    mask = np.zeros((height, width), np.int8)
    cv2.fillConvexPoly(mask, hull, 1, lineType=8, shift=0)
    mask = mask.astype(np.bool)
    return mask


def generate_random_colors(num):
    """
    Return a list of random colors in rgb format.
    """
    colors = cmap_d["jet"](np.linspace(0, 1, num))
    colors = [[int(255*c) for c in color[:3]] for color in colors]
    return colors


def get_video_timestamps(video_file):
    """
    Get the timestamps of the frames in a video stream.
    Reuturn a list of integers of length N+1 where N is
    the total number of frames of the video and the i-th
    item in the list is the start time of the i-th frame
    in miliseconds.

    Example:
    >>> get_video_timestamps("test.mp4")
    array([0, 34, 67, 101, 134, ..., ])
    """
    stream = cv2.VideoCapture(video_file)
    num_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    stream.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    total_time = stream.get(cv2.CAP_PROP_POS_MSEC)
    time_ms_float = np.linspace(0, total_time, num_frames + 1)
    return np.around(time_ms_float).astype("int")


def ffmpeg_pipe(width, height):
    ffmpeg = subprocess.Popen(("ffmpeg",
                               "-threads", "0",
                               "-loglevel", "panic",
                               "-r", "%d" % 30,
                               "-f", "rawvideo",
                               "-pix_fmt", "bgr24",
                               "-s", "%dx%d" % (width, height),
                               "-i", "-",
                               "-c:v", "libx264",
                               "-crf", "20",
                               "-y",  "trajectory_result.mp4"),
                              stdin=subprocess.PIPE)
    return ffmpeg.stdin


def check_shape(arr, shape, desc_str):
    """
    Check if a numpy ndarray has an expected shape.
    """
    arr = np.asarray(arr)
    if arr.shape != shape:
        raise ValueError("An array of shape {} is expected for {} \
but the input has shape {}.".format(shape, desc_str, arr.shape))

    return arr


def column_vector(arr, dim=None, desc_str=None):
    """
    Convert an input array to a (dim, 1) column vector.
    """
    arr = np.atleast_2d(arr)
    if arr.shape[0] == 1:
        arr = arr.T

    if dim is None:
        dim = arr.shape[0]

    if arr.shape != (dim, 1):
        err_info = "cannot convert array to a vector of shape ({}, 1)"
        if desc_str:
            err_info += " for {}"
        raise ValueError(err_info.format(dim, desc_str))

    return arr


def array_from_csv(string):
    """
    Convert a string from csv to numpy array.

    Example:
    >>> string = "[1 1 1 1]"
    >>> arr = array_from_csv(string)
    >>> arr
    array([1, 1, 1, 1])
    """
    arr = np.fromstring(string[1: -1], sep=" ", dtype=int)

    if arr.size == 0:
        return None

    return arr


def write_namedtuples_csv(cls, csvfile, namedtuples_list):
    """
    Write the data of a list of namedtuples to a csv file.
    """
    with open(csvfile, "w") as f_csv:
        # directly use the fields of the namedtuple class
        writer = csv.DictWriter(f_csv, fieldnames=cls._fields, lineterminator="\n")
        writer.writeheader()
        for obj in namedtuples_list:
            # call the `_asdict()` method of the namedtuple
            writer.writerow(obj._asdict())


def read_namedtuples_csv(cls, csvfile):
    """
    Read the data of a list of namedtuples from a csv file.
    """
    namedtuples_list = []
    with open(csvfile, "r") as f:
        f_csv = csv.DictReader(f, lineterminator="\n")
        for row in f_csv:
            data = {}
            for field_name, type_converter in cls._field_types.items():
                if field_name in row:
                    data[field_name] = type_converter(row[field_name])

            namedtuples_list.append(cls(**data))
    return namedtuples_list


def bbox_to_xywh(bbox):
    x1, y1, x2, y2 = [int(x) for x in bbox]
    return (x1, y1, x2 - x1, y2 - y1)


def xywh_to_bbox(xywh):
    x, y, w, h = [int(x) for x in xywh]
    return (x, y, x + w, y + h)


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def iou_rect(rect1, rect2):
    """
    Compute the iou score of two rectangles.
    """
    xA = max(rect1[0], rect2[0])
    yA = max(rect1[1], rect2[1])
    xB = min(rect1[2], rect2[2])
    yB = min(rect1[3], rect2[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    area1 = (rect1[2] - rect1[0] + 1) * (rect1[3] - rect1[1] + 1)
    area2 = (rect2[2] - rect2[0] + 1) * (rect2[3] - rect2[1] + 1)
    return inter / (area1 + area2 - inter)

    
def mask_rect_diff(mask, rect):
    """
    Compute the intersection and difference sets of two a mask and bounding box.
    """
    x1, y1, x2, y2 = rect
    B = mask[y1: y2+1, x1: x2+1]
    overlap = np.count_nonzero(B)
    diff1 = np.count_nonzero(mask) - overlap
    diff2 = (y2 - y1 + 1) * (x2 - x1 + 1) - overlap
    return [overlap, diff1, diff2]


def iou_det_object(detobj1, detobj2):
    """
    Compute the iou score of two `DetObject`s.
    """
    return iou_rect(detobj1.box2d, detobj2.box2d)


def write_trajectories_csv(trajectory_list, csvfile):
    with open(csvfile, "w") as f_csv:
        writer = csv.writer(f_csv, lineterminator="\n")
        for traj in trajectory_list:
            di = traj._asDict()
            for key, value in di.items():
                writer.writerow([key, value])
            
