import numpy as np
import cv2


__all__ = ["draw_NED_frame", "draw_mask_on_image", "draw_reproj_point_pairs",
           "draw_box3d_on_image", "draw_camera_anchor_points", "draw_detobj_on_image",
           "draw_zone_on_image", "draw_track_on_image", "draw_trajectory_on_image",
           "draw_calibation_annotations"]


def draw_NED_frame(camera, image, length=10):
    """
    Draw a (North, East) coordinate frame on an image.
    """
    L = length

    # origin, x-axis, y-axis, z-axis in meters in world frame
    points = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]],
                      dtype=np.float32)

    # project them to image plane
    pixels, jacobian = cv2.projectPoints(points, *camera.params)

    # fetch their pixels
    pt_o, pt_x, pt_y, pt_z = np.around(pixels).astype(np.int).reshape(-1, 2)

    # draw the axis
    cv2.line(image, (pt_o[0], pt_o[1]), (pt_x[0], pt_x[1]), (0, 255, 0), 2)
    cv2.line(image, (pt_o[0], pt_o[1]), (pt_y[0], pt_y[1]), (255, 0, 0), 2)
    cv2.line(image, (pt_o[0], pt_o[1]), (pt_z[0], pt_z[1]), (0, 0, 255), 2)

    # draw labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "O", (pt_o[0], pt_o[1] + 30), font,
                1, (0, 0, 255), 2)
    cv2.putText(image, "N", (pt_x[0], pt_x[1] + 30), font,
                1, (0, 255, 0), 2)
    cv2.putText(image, "E", (pt_y[0], pt_y[1] + 30), font,
                1, (255, 0, 0), 2)
    cv2.putText(image, "Z", (pt_z[0], pt_z[1] + 30), font,
                1, (0, 0, 255), 2)
    return image


def draw_mask_on_image(image, mask, color=(0, 0, 255)):
    """
    Paint the region defined by a given mask on an image.
    """
    new_image = np.zeros_like(image)
    new_image[:, :] = color
    mask = np.array(mask, dtype=np.uint8)
    new_mask = cv2.bitwise_and(new_image, new_image, mask=mask)
    cv2.addWeighted(image, 1.0, new_mask, 0.5, 0.0, image)
    return image


def draw_reproj_point_pairs(image, pixels, pixels_reproj):
    """
    Display a list of pixels and their reprojected counterparts
    on an image simultaneously.
    """
    # draw original pixels in yellow
    for x, y in pixels:
        cv2.circle(image, (int(x), int(y)), 6, (0, 255, 255), -1)

    # draw re-projected pixels in red
    for x, y in pixels_reproj:
        cv2.circle(image, (int(x), int(y)), 6, (0, 0, 255), -1)
    return image


def draw_box3d_on_image(box3d, image, camera, color=(255, 0, 0),
                        thickness=2, front_color=(255, 0, 255)):
    """
    Display a `Box3D` object on an image.
    """
    # project the 8 corners of `box3d` to image plane
    pixels = camera.world_to_image(box3d.corners).reshape(-1, 2)
    pixels = tuple(tuple(p) for p in pixels)

    # draw the corners of the projected box
    for px in pixels:
        cv2.circle(image, px, 3, color, -1)

    # indices of edges not belong to the front face
    non_front_indices = [(4, 5), (6, 7), (4, 6), (5, 7),
                         (2, 6), (3, 7), (0, 4), (1, 5)]
    # indices of egdes belong to the front face
    front_indices = [(0, 1), (0, 2), (1, 3), (2, 3)]

    for i, j in non_front_indices:
        cv2.line(image, pixels[i], pixels[j], color, thickness)

    # highlight the front face of the 3d box
    for i, j in front_indices:
        cv2.line(image, pixels[i], pixels[j], front_color, thickness)
    return image


def draw_camera_anchor_points(camera, image, color=(0, 0, 255)):
    if camera.anchor_points is not None:
        pixels = camera.world_to_image(camera.anchor_points)
        for x, y in pixels:
            cv2.circle(image, (x, y), 6, color, -1)

        coords = np.around(camera.anchor_points, 2)[:, :2]
        for pix, pt in zip(pixels, coords):
            x0, y0 = pix
            x1, y1 = pt
            text = "({}, {})".format(x1, y1)
            cv2.putText(image, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image


def draw_calibation_annotations(image):
    text1 = "yellow: input pixels"
    cv2.putText(image, text1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    text2 = "red: reprojected pixels"
    cv2.putText(image, text2, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    text3 = "blue: test points"
    cv2.putText(image, text3, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return image


def draw_detobj_on_image(detobj, image, color=(255, 0, 0), custom_text=None):
    """
    Display a `DetObject` on an image.
    """
    # draw bounding box
    x1, y1, x2, y2 = detobj.box2d

    # add text
    if custom_text:
        text = custom_text
    else:
        text = "{}: {:.3f}".format(detobj.label, detobj.score)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.6
    tex_width = cv2.getTextSize(text, font, fontscale, 2)[0][0]
    # set text location
    xl = x1 if tex_width < x2 - x1 else x1 - (tex_width - x2 + x1) // 2
    xr = xl + tex_width + 10
    cv2.rectangle(image, (xl, y1 - 30), (xr, y1), (255, 255, 255), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(image, (xl, y1 - 30), (xr, y1), color, 2)

    y = y1 - 10 if y1 > 20 else y1 + 10
    cv2.putText(image, text, (xl + 5, y), font, fontscale, color, 2)

    return image


def draw_zone_on_image(zone, image, color=(0, 0, 255)):
    pixels = zone.hull_im.reshape(-1, 1, 2).astype(int)
    cv2.polylines(image, [pixels], True, color, 2)


def draw_track_on_image(track, frame_index, frame, state, color):
    detobj = track.get_history_detection_by_frame(frame_index)
    x, y, v = np.round(state[:3, 0], decimals=2)
    custom_text = "{} | id:{}, pos: ({}, {}), {}m/s".format(
        track.agent_type,
        track.track_id,
        x,
        y,
        abs(v)
    )

    if detobj is not None:
        draw_detobj_on_image(
            detobj,
            frame,
            color,
            custom_text)

    return frame


def draw_trajectory_on_image(traj, frame_index, frame, zone, color):
    """
    Draw the history of a trajectory up to a given frame on an image.
    """
    # draw the dotted history trajectory.
    for k in range(frame_index + 1):
        index = traj.track.get_history_index_by_frame(k)
        if index is not None:
            pixel = traj.pixel_history[index]
            if zone.in_zone(pixel):
                x, y = [int(c) for c in pixel]
                cv2.circle(frame, (x, y), 2, color, -1)

    # draw the label of the track
    index = traj.track.get_history_index_by_frame(frame_index)
    if index is not None:
        _, state, _ = traj.trajectory[index]
        draw_track_on_image(traj.track, frame_index, frame, state, color)
