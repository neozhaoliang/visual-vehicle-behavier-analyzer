import cv2
import tqdm
from .conv_nets import YOLOV3_Net
from .det_object import DetObject
from .track import TrackDetAssociation, CVTrackMerger
from utils.visualize import draw_detobj_on_image
from utils.gui import display_image


# ---------------------
# object colors used in YOLO detection window
colors = {"person":    [0,   255, 255],
          "car":       [255, 255, 0],
          "bus":       [255, 0,   255],
          "truck":     [127, 150, 255],
          "bicycle":   [255, 100, 127],
          "motorbike": [100, 255, 127]}


def get_tracks_from_video(video_file, net, zone=None, print_track=True):
    stream = cv2.VideoCapture(video_file)
    # track-detection association
    asso = TrackDetAssociation(iou_threshold=0.2)
    # process bar
    bar = tqdm.tqdm(desc="processing video frames",
                    total=int(stream.get(cv2.CAP_PROP_FRAME_COUNT)))
    # ---------------
    # run yolov3 detection
    frame_index = 0
    while True:
        success, frame = stream.read()
        if not success:
            break

        # get detections
        detobjs_list = net.get_inference(frame, frame_index)
        for detobj in detobjs_list:
            draw_detobj_on_image(detobj, frame, color=colors[detobj.label])

        # update tracks
        asso.update(detobjs_list)
        frame_index += 1
        bar.update(1)

        ret = display_image("YOLOV3", frame)
        if ret < 0:
            break

    stream.release()
    cv2.destroyAllWindows()
    bar.close()

    # track-track merger.
    newstream = cv2.VideoCapture(video_file)
    merger = CVTrackMerger(newstream)
    final_tracks = merger.run_merge_tracks(asso.tracks, zone)
    for track in final_tracks:
        print("track {}: (start frame: {:04d}, end frame: {:04d})".format(
            track.track_id,
            track.init_frame_index,
            track.last_frame_index))

    newstream.release()
    return final_tracks
