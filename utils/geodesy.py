import numpy as np
import yaml


LOCATION_SCALING_FACTOR = 111318.84502145034


def longitude_scaling(lat):
    return np.cos(np.radians(lat))


def _get_NE_offsets(origin, point):
    """
    Compute (north, east) offsets of a point relative to a given origin.
    Both `point` and `origin` are of the form (latitude, longitude).
    """
    lat0, lon0 = origin
    lat1, lon1 = point
    N = (lat1 - lat0) * LOCATION_SCALING_FACTOR
    E = (lon1 - lon0) * LOCATION_SCALING_FACTOR * longitude_scaling(lat0)
    return (N, E)


def convert_latlon_cartesian(origin, points):
    """
    Compute (north, east) offsets of a list of (lat, lon) points
    relative to a given origin. The returned coordinates are in
    3d space with their z-components are all zeros.
    """
    origin = np.asarray(origin).reshape(2)
    points = np.asarray(points).reshape(-1, 2)
    n = len(points)
    result = np.zeros((n, 3), dtype="double")
    result[:, :2] = [_get_NE_offsets(origin, pt) for pt in points]
    return result


def load_scene_yaml(yamlfile, convert_cartesian=True):
    """
    Load world and pixel points data from an user input yaml file.
    """
    def check_yaml_data(data, key, required=True):
        value = data.get(key, None)
        if value is None:
            if required:
                raise ValueError("missing '{}' field in the scene yaml \
file".format(key))
            else:
                return None
        return np.array(value, dtype=np.float)

    with open(yamlfile, "r") as f_yaml:
        data = yaml.load(f_yaml, Loader=yaml.SafeLoader)
        image_pixels = check_yaml_data(data, "image-pixels")
        world_origin = check_yaml_data(data, "world-origin")
        world_points = check_yaml_data(data, "world-points")
        test_points = check_yaml_data(data, "test-points", required=False)

    if convert_cartesian:
        world_points = convert_latlon_cartesian(world_origin, world_points)
        if test_points is not None:
            test_points = convert_latlon_cartesian(world_origin, test_points)

    return image_pixels, world_origin, world_points, test_points
