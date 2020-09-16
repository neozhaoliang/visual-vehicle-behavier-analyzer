import os
import cv2
import tqdm
import numpy as np
from analyzer.conv_nets import YOLOV3_Net
from analyzer.camera import PinholeCamera
from analyzer.trajectory import TrajectoryProcessor
from analyzer.zone import Zone
from utils import (get_video_timestamps, ffmpeg_pipe,
                   generate_random_colors, write_trajectories_csv)
from utils.visualize import (draw_detobj_on_image, draw_NED_frame,
                             draw_zone_on_image, draw_trajectory_on_image)
from utils.gui import display_image
from analyzer import get_tracks_from_video


# current directory
work_dir = os.getcwd()
# yolov3 directory
yolov3_coco_dir = os.path.join(work_dir, "yolov3_coco")
# scene data directory
scene_data_dir = os.path.join(work_dir, "scene_data")
# input video stream
video_file = os.path.join(scene_data_dir, "test.mp4")
# camera yaml file
camera_file = os.path.join(scene_data_dir, "camera_params.yaml")
# zone yaml file
zone_file = os.path.join(scene_data_dir, "zone.yaml")


def main():
    # video timestamps
    list_times_ms = get_video_timestamps(video_file)
    # pinhole camera model
    camera = PinholeCamera.read_yaml(camera_file)
    # get zone
    zone = Zone.read_yaml(zone_file)
    # yolov3 net
    net = YOLOV3_Net(yolov3_coco_dir, use_gpu=False)
    # detect tracks in the video
    tracks = get_tracks_from_video(video_file, net, zone)

    # trajectory processors
    trajectories = []
    for track in tracks:
        traj = TrajectoryProcessor(track, camera)
        traj.process_trajectory(list_times_ms)
        trajectories.append(traj)

    rand_colors = generate_random_colors(len(tracks))
    count = 0
    stream = cv2.VideoCapture(video_file)
    pipe = None
    print("[INFO] Drawing trajectories on frames ...")
    while True:
        success, frame = stream.read()
        if not success:
            break

        cv2.putText(frame, "frame: {}".format(count), (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # diplay NE frame
        draw_NED_frame(camera, frame)
        # display interested zone
        draw_zone_on_image(zone, frame)

        for traj, color in zip(trajectories, rand_colors):
            draw_trajectory_on_image(traj, count, frame, zone, color)

        ret = display_image("Vehicle Trajectories", frame)
        if ret < 0:
            break

        if pipe is None:
            pipe = ffmpeg_pipe(*frame.shape[:2][::-1])

        pipe.write(frame.tobytes())
        count += 1

    cv2.destroyAllWindows()
    stream.release()
    pipe.close()
    # write trajectory data to csv file
    write_trajectories_csv(trajectories, "trajectories.csv")


if __name__ == "__main__":
    main()
