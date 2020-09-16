import cv2
import numpy as np
from scipy.spatial import ConvexHull


class Zone(object):

    """
    Class for the zone that we are interested.
    """

    def __init__(self, pixels, world_points):
        """
        pixels: a list of pixels determine the zone.
        world_points: the corresponding world points of the pixels.
        """
        self.pixels = pixels.astype(int)
        self.world_points = world_points[:, :2].astype(int)
        self.hull_im = cv2.convexHull(self.pixels)
        self.hull_world = cv2.convexHull(self.world_points)

    @classmethod
    def read_yaml(cls, yamlfile):
        fs = cv2.FileStorage(yamlfile, cv2.FileStorage_READ)
        pixels = fs.getNode("zone-image").mat().reshape(-1, 2)
        points = fs.getNode("zone-world").mat().reshape(-1, 3)
        return cls(pixels, points)

    def in_zone(self, pt, image=True):
        """
        Check if the location of a point is in this zone.
        """
        xy = (int(pt[0]), int(pt[1]))
        if image:
            contour = self.pixels
        else:
            contour = self.world_points

        ret = cv2.pointPolygonTest(contour, xy, False)
        return ret >= 1

    @classmethod
    def from_scene(cls, gui, camera, output_file):
        pixels = np.int32(gui.keypoints)
        hull_image = cv2.convexHull(pixels)
        points = camera.image_to_world(pixels)
        vert_inds = ConvexHull(points[:, :2]).vertices
        hull_world = points[vert_inds]
        fs = cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE)
        fs.write("zone-image", hull_image)
        fs.write("zone-world", hull_world)
        fs.release()
