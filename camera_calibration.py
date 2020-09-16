import os
import cv2
from analyzer.camera import PinholeCamera
from analyzer.zone import Zone
from utils.gui import display_image, PointSelector


# require: an input yaml file and a frame from the video.
# the input yaml file contains the GPS data of the selected points
# and their pixel locations in the video.
data_dir = os.path.join(os.getcwd(), "scene_data")
scene_image = os.path.join(data_dir, "example_scene.png")
scene_yaml = os.path.join(data_dir, "example_scene.yaml")
zone_yaml = os.path.join(data_dir, "zone.yaml")


def main():
    image, camera = PinholeCamera.from_scene(scene_image, scene_yaml)
    while True:
        ret = display_image("camera caliberation", image)
        if ret < 0:
            print("exit without saving the camera parameters")
            break

        if ret == 1:
            print("saving camera parameters to yaml file")
            camera.save_yaml(os.path.join(data_dir, "camera_params.yaml"))
            cv2.imwrite(os.path.join(data_dir, "camera_calibration.png"), image)

    cv2.destroyAllWindows()

    gui = PointSelector(image, title="select points to choose the zone")
    ret = gui.loop()
    if ret:
        Zone.from_scene(gui, camera, zone_yaml)
        print("write zone data to yaml file")


if __name__ == "__main__":
    main()
