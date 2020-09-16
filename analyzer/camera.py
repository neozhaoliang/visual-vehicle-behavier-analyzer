"""
Camera calibration and convertion between image pixels and world points.
"""
import numpy as np
from numpy.linalg import inv
import cv2
from scipy.optimize import minimize
from utils import check_shape, column_vector
from utils.geodesy import load_scene_yaml
from utils.visualize import (draw_NED_frame,
                             draw_reproj_point_pairs,
                             draw_camera_anchor_points,
                             draw_calibation_annotations)


class PinholeCamera(object):

    """
    Ideal pinhole camera model class.

    Reference:

    https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    """

    def __init__(self, rotMatrix, transVect, cameraMatrix,
                 distCoeffs, anchor_points=None):
        """
        Extrinsic parameters : rotMatrix, transVect.
        Intrinsic parameters : cameraMatrix, distCoeffs.
        """
        self.rotMatrix = check_shape(rotMatrix, (3, 3), "rotation matrix")
        self.transVect = column_vector(transVect, 3, "translation vector")
        self.cameraMatrix = check_shape(cameraMatrix, (3, 3), "camera matrix")
        self.distCoeffs = column_vector(distCoeffs, 4, "distort coefficients")
        self.anchor_points = anchor_points

    @property
    def params(self):
        return (self.rotMatrix, self.transVect,
                self.cameraMatrix, self.distCoeffs)

    @classmethod
    def read_yaml(cls, filename):
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        rotMatrix = fs.getNode("rotMatrix").mat()
        transVect = fs.getNode("transVect").mat()
        cameraMatrix = fs.getNode("cameraMatrix").mat()
        distCoeffs = fs.getNode("distCoeffs").mat()
        fs.release()
        return cls(rotMatrix, transVect, cameraMatrix, distCoeffs)

    def save_yaml(self, filename):
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        fs.write("rotMatrix", self.rotMatrix)
        fs.write("transVect", self.transVect)
        fs.write("cameraMatrix", self.cameraMatrix)
        fs.write("distCoeffs", self.distCoeffs)
        fs.release()

    def world_to_image(self, world_points):
        """
        Project a list of points in world frame to pixels.
        The input is converted to a nx3 numpy 2d array,
        The output is a nx2 numpy 2d array of int type.
        """
        points = np.array(world_points, dtype="double").reshape(-1, 3)
        pixels, jacobian = cv2.projectPoints(points, *self.params)
        pixels = np.around(pixels).astype(np.int).reshape(-1, 2)
        return pixels

    def image_to_world(self, image_pixels, z0=0):
        """
        Project a list of pixels to the z=z0 plane in world frame.
        """
        R, T, K, D = self.params
        Kinv = inv(K)
        Rinv = R.transpose()
        RinvT = np.dot(Rinv, T)
        new_col = np.ones((len(image_pixels), 1))
        pixels = np.append(image_pixels, new_col, axis=1).reshape(-1, 3, 1)
        result = []
        for pixel in pixels:
            pt = Rinv.dot(Kinv.dot(pixel))
            z = z0 + RinvT[2, 0]
            pt = pt * (z / pt[2, 0])
            pt -= RinvT
            result.append(pt)

        return np.float64(result).reshape(-1, 3)

    def get_jacobi_matrix(self, world_point):
        """
        Given a point `q` in world frame and its corresponding pixel `Pix`,
        return the jacobi matrix of Pix = f(q) at q.
        """
        R, T, K, D = self.params
        # Pix = (x, y, 1), sPix = s * Pix = K * (R * q + T), s = sPix[2, 0]
        q = column_vector(world_point, 3)
        sPix = K.dot(R.dot(q) + T)
        s = sPix[2, 0]
        Pix = sPix / s

        # d(sPix)/dq = K * R
        # d(sPix)/dq = (d(s*x)/dq, d(s*y)/dq, ds/dq) ==> ds/dq = (K*R)[2]
        # d(sPix)/dq = ds/dq * Pix + dPix/dq * s ==
        # ==> dPix/dq = (d(sPix)/dq - ds/dq * Pix) / s
        dsPix = np.dot(K, R)
        ds = dsPix[2].reshape(1, 3)
        dPix = (dsPix - np.dot(Pix,  ds)) / s
        return dPix[:2]

    @classmethod
    def from_scene(cls, scene_image, scene_yaml, draw_image=True):
        return get_camera_from_scene_calib(scene_image, scene_yaml, draw_image)

    def __str__(self):
        return ("Pinhole Camera:\n" +
                "rotMatrix: {}\n".format(self.rotMatrix) +
                "transVect: {}\n".format(self.transVect) +
                "cameraMatrix: {}\n".format(self.cameraMatrix) +
                "distCoeffs: {}\n".format(self.distCoeffs))


def _get_camera_params_simple(focal_length, image_shape, image_pixels,
                              world_points):
    """
    A direct estimation of the camera extrinsic/intrinsic parameters
    with known focal length.
    """
    height, width = image_shape[:2]
    cy, cx = (height / 2, width / 2)

    # a guess of the intrinsic matrix as an ideal pinhole camera.
    cameraMatrix = np.array([[focal_length, 0,            cx],
                             [0,            focal_length, cy],
                             [0,            0,            1 ]],
                            dtype=np.float64)

    # the distort coefficients are assumed all zeros
    distCoeffs = np.zeros((4, 1))

    # estimate rotation matrix and translation vector
    success, rotVect, transVect = cv2.solvePnP(
        world_points,
        image_pixels,
        cameraMatrix,
        distCoeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)
    rotMatrix, _ = cv2.Rodrigues(rotVect)

    return rotMatrix, transVect, cameraMatrix, distCoeffs


def _get_pinhole_camera_opt(image_shape, image_pixels, world_points):
    """
    Estimate camera parameters without knowing the focal length by calling
    `scipy.optimize.minimize` to minimize the mean square error between
    original pixels and the reprojected ones.
    """

    def func_constraint(params):
        """
        Constraint on the parameters to be estimated, it must be positive.
        """
        focal_length, = params
        return focal_length

    def func_cost(params, image_shape, image_pixels, world_points):
        """
        Function to be optimized, its' the mean square error between the
        original pixels and the reprojected ones.
        """
        focal_length, = params
        rotMatrix, transVect, cameraMatrix, distCoeffs = \
            _get_camera_params_simple(
                focal_length,
                image_shape,
                image_pixels,
                world_points)

        pixels_reproj, jacobian = cv2.projectPoints(
            world_points,
            rotMatrix,
            transVect,
            cameraMatrix,
            distCoeffs)
        pixels_reproj = pixels_reproj[:, 0]

        return np.linalg.norm(np.subtract(image_pixels, pixels_reproj))

    height, width = image_shape[:2]
    p0 = (width,)  # initial guess of the focal length
    cons = {"type": "ineq", "fun": lambda x: func_constraint(x)}
    params = minimize(
        func_cost,
        p0,
        constraints=cons,
        args=(image_shape, image_pixels, world_points))

    focal_length = params.x[0]
    rotMatrix, transVect, cameraMatrix, distCoeffs = \
        _get_camera_params_simple(
            focal_length,
            image_shape,
            image_pixels,
            world_points)

    return PinholeCamera(rotMatrix, transVect, cameraMatrix,
                         distCoeffs, anchor_points=world_points)


def get_camera_from_scene_calib(scene_image, scene_yaml, draw_image=True):
    """
    Estimate camera parameters from a scene image and a yaml file.
    The yaml file must contain the (lat, lon) world points and their
    corresponding pixels in the scene image.

    scene_image : str, path to the scene image.
    scene_yaml : str, path to the scene yaml file.
    """
    image_pixels, world_origin, world_points, test_points = \
        load_scene_yaml(scene_yaml)

    # read the static scene image taken by the camera (usually the first
    # frame of the video)
    image = cv2.imread(scene_image)
    if image is None:
        raise ValueError("Failed to load image: {}".format(scene_image))

    # estimate the camera parameters and get the camera model
    camera = _get_pinhole_camera_opt(image.shape, image_pixels, world_points)

    if draw_image:
        # project the world points and test points to image for comparison
        pixels_reproj = camera.world_to_image(world_points)

        # draw the NED frame
        draw_NED_frame(camera, image)

        # draw the pixels and reprojected pixels
        draw_reproj_point_pairs(image, image_pixels, pixels_reproj)

        # add text annotations of the coordinates
        draw_camera_anchor_points(camera, image, color=(0, 0, 255))

        # add text annotations of the colors
        draw_calibation_annotations(image)

        # draw the test points
        if test_points is not None:
            pixels_test = camera.world_to_image(test_points)
            for pt in pixels_test:
                cv2.circle(image, (pt[0], pt[1]), 8, (255, 0, 0), 2, -1)

    return image, camera
