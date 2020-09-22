"""
Example usage:

>>> net = YOLOV3_Net("yolo_coco", use_gpu=True)
>>> detections = net.get_inference(frame, frame_index)

Here the detections are stored in a list and each item is a
namedtuple 'DetObject'.
"""
import os
import cv2
import numpy as np

from .det_object import DetObject


INTERESTED_OBJECTS = ["person", "bicycle", "car",
                      "motorbike", "bus", "truck"]


class YOLOV3_Net(object):

    CONFIDENCE_THRESHOLD = 0.5
    OVERLAPPING_THRESHOLD = 0.2

    def __init__(self, yolo_coco_dir, use_gpu=False):
        # read coco labels
        coco_labels_file = os.path.join(yolo_coco_dir, "coco_labels.txt")
        with open(coco_labels_file, "r") as f:
            self.class_names = f.read().strip().split("\n")

        # load model
        config_file = os.path.join(yolo_coco_dir, "yolov3.cfg")
        weights_file = os.path.join(yolo_coco_dir, "yolov3.weights")
        self.net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        layers = self.net.getLayerNames()
        self.layers = [layers[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def get_inference(self, frame, frame_index):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        r = self.net.forward(self.layers)
        H, W = frame.shape[:2]
        return self._pack_frame_detections(r, frame_index, W, H)

    def _pack_frame_detections(self, r, frame_index, W, H):
        boxes = []
        confidences = []
        class_ids = []
        for output in r:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                score = scores[class_id]
                if score > self.CONFIDENCE_THRESHOLD:
                    bbox = detection[:4] * np.array([W, H, W, H])
                    cx, cy, width, height = bbox
                    x1 = int(cx - width / 2)
                    y1 = int(cy - height / 2)
                    boxes.append([x1, y1, int(width), int(height)])
                    confidences.append(round(float(score), 2))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD,
                                self.OVERLAPPING_THRESHOLD)

        detobj_list = []
        if len(idxs) > 0:
            for det_id in idxs.flatten():
                label = self.class_names[class_ids[det_id]]
                if self.interested(label):
                    x, y, w, h = boxes[det_id]
                    bbox = np.int32([x, y, x + w, y + h])
                    score = confidences[det_id]
                    detobj = DetObject(
                        frame_index,
                        det_id,
                        label,
                        score,
                        bbox,
                        np.int32([W, H])
                    )
                    detobj_list.append(detobj)

        return detobj_list

    def interested(self, label):
        return label in INTERESTED_OBJECTS
