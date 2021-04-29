# 单目纯视觉分析车辆行为

这个项目演示通过用一部手机，拍摄道路上的一段交通视频，可以提取哪些有用的信息。

运行方法：

1. 首先将 YOLOv3 的权重文件 [yolov3.weights](https://pjreddie.com/darknet/yolo/) 下载并放在 `yolov3_coco` 目录下。
     ```
     wget https://pjreddie.com/media/files/yolov3.weights
     ```
2. 运行 `python main.py` 查看效果。如果你的 Opencv 有 cuda 加速支持的话可以在 `main.py` 中将 `use_gpu` 设置为 `True`。

详细文档见 `doc` 目录。

> 修复了 opencv 版本不同导致的轨迹不正确的错误。