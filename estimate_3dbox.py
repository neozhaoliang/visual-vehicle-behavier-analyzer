import os
import cv2
from analyzer.box3d import get_box3d_from_detobj, Box3D
from analyzer.det_object import DetObject
from analyzer.camera import PinholeCamera
from analyzer.conv_nets import YOLOV3_Net
from utils.visualize import draw_box3d_on_image
from utils.gui import display_image


def main():
    cwd = os.getcwd()
    # prepare yolov3 net
    yolo_dir = os.path.join(cwd, "yolov3_coco")
    net = YOLOV3_Net(yolo_dir)
    # prepare pinhole camera
    scene_dir = os.path.join(cwd, "scene_data")
    camera_file = os.path.join(scene_dir, "camera_params.yaml")
    camera = PinholeCamera.read_yaml(camera_file)
    # prepare image to detect
    example_image = os.path.join(scene_dir, "example_scene.png")
    image = cv2.imread(example_image)
    # get inference
    detobj_list = net.get_inference(image, 0)
    box3d_list = [get_box3d_from_detobj(detobj, camera) for detobj in detobj_list]
    for box3d in box3d_list:
        draw_box3d_on_image(box3d, image, camera)

    while True:
        ret = display_image("estimate 3D box of vehicles", image)
        if ret < 0:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
