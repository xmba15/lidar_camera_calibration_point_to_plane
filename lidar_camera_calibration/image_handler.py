#!/usr/bin/env python
import json
import logging
from typing import Optional

import cv2
import numpy as np

from .types import CameraInfo, Plane

__all__ = ["ImageHandler"]

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s:%(message)s")
_logger = logging.getLogger(__name__)


_BLUE = (255, 0, 0)
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)


class ImageHandler:
    def __init__(
        self,
        checkerboard_rows: int,
        checkerboard_cols: int,
        checkerboard_size: float,
        debug: bool,
        camera_info: CameraInfo,
    ):
        self._checkerboard_rows = checkerboard_rows
        self._checkerboard_cols = checkerboard_cols
        self._checkerboard_size = checkerboard_size
        self._debug = debug
        self._camera_info = camera_info

        self._object_points = self._create_object_points()

    @property
    def camera_info(self):
        return self._camera_info

    def run(self, image_file: str) -> Optional[Plane]:
        gray_image = cv2.imread(image_file, 0)
        assert gray_image is not None, f"failed to read gray_image: {image_file}"
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        found, corners = cv2.findChessboardCorners(
            gray_image,
            (self._checkerboard_cols, self._checkerboard_rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            _logger.info(f"failed to extract corners of {image_file}")
            return None
        cv2.cornerSubPix(
            gray_image, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        )
        _, rvec, tvec = cv2.solvePnP(
            self._object_points,
            corners,
            self._camera_info.K,
            self._camera_info.dist_coeffs,
            None,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if self._debug:
            import os

            image = cv2.imread(image_file)
            image_file_name = os.path.join("/tmp", image_file.split("/")[-1])
            axis = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0]]).reshape(-1, 3)
            axis_points, _ = cv2.projectPoints(axis, rvec, tvec, self._camera_info.K, self._camera_info.dist_coeffs)
            image = self._draw_pose(image, axis_points)
            cv2.imwrite(image_file_name, image)

        rot_mat, _ = cv2.Rodrigues(rvec)
        normal_v = rot_mat[:, 2]
        D = -normal_v.dot(tvec)
        plane_coefficients = np.array([*normal_v, D], dtype=float)

        return Plane(plane_coefficients)

    def _draw_pose(self, image, axis_points):
        axis_points = axis_points.astype(int)
        image = cv2.line(
            image,
            tuple(axis_points[3].ravel()),
            tuple(axis_points[0].ravel()),
            _RED,
            3,
        )
        image = cv2.line(
            image,
            tuple(axis_points[3].ravel()),
            tuple(axis_points[1].ravel()),
            _GREEN,
            3,
        )
        image = cv2.line(
            image,
            tuple(axis_points[3].ravel()),
            tuple(axis_points[2].ravel()),
            _BLUE,
            3,
        )
        return image

    def _create_object_points(self):
        object_points = np.zeros((self._checkerboard_rows * self._checkerboard_cols, 3))
        for i in np.arange(self._checkerboard_rows):
            for j in np.arange(self._checkerboard_cols):
                object_points[i * self._checkerboard_cols + j, :2] = np.array((i, j)) * self._checkerboard_size

        return object_points

    @staticmethod
    def from_json(dataset_info_json: str) -> "ImageHandler":
        with open(dataset_info_json, "r") as _file:
            data = json.load(_file)
            for key in ("checkerboard_rows", "checkerboard_cols", "checkerboard_size", "debug"):
                assert key in data

        return ImageHandler(
            data["checkerboard_rows"],
            data["checkerboard_cols"],
            data["checkerboard_size"],
            data["debug"],
            CameraInfo.from_json(dataset_info_json),
        )
