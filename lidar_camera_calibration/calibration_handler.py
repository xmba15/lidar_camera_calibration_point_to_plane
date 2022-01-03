#!/usr/bin/env python
import logging
from typing import Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .cloud_handler import CloudHandler
from .data_loader import DataLoader
from .image_handler import ImageHandler
from .point_to_plane_optimization import optimize as optimize_pp
from .types import Array

try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal

__all__ = ["CalibrationHandler"]

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s:%(message)s")
_logger = logging.getLogger(__name__)


_GREEN = (0, 255, 0)


class CalibrationHandler:
    def __init__(self, dataset_info_json: str, image_lists_path: str, cloud_lists_path: str, data_path: str):
        self._data_loader = DataLoader(image_lists_path, cloud_lists_path, data_path)
        self._cloud_handler = CloudHandler.from_json(dataset_info_json)
        self._image_handler = ImageHandler.from_json(dataset_info_json)

    def run(self):
        all_planes_lidar = list()
        all_planes_camera = list()

        for (image_file, cloud_file) in zip(tqdm(self._data_loader.all_image_files), self._data_loader.all_cloud_files):
            plane_camera = self._image_handler.run(image_file)
            if plane_camera is None:
                continue
            plane_lidar = self._cloud_handler.run(cloud_file)
            if plane_lidar is None:
                continue

            all_planes_camera.append(plane_camera)
            all_planes_lidar.append(plane_lidar)

        if len(all_planes_lidar) == 0:
            _logger.warning("failed to extract planes data")
            return None

        pose_SE3 = optimize_pp(all_planes_lidar, all_planes_camera)
        transformation_mat = pose_SE3.as_matrix()

        rpy = np.deg2rad(cv2.RQDecomp3x3(transformation_mat[:3, :3])[0])
        tvec = transformation_mat[:3, 3]

        _logger.info("drawing projected lidar points on image...")
        self._draw_projections_lidar_on_cam(transformation_mat)

        _logger.info(f"rpy rotation: {np.rad2deg(rpy)}[degree]")
        _logger.info(f"translation: {tvec}[meter]")

    def _draw_projections_lidar_on_cam(self, transformation_mat):
        import os

        for (image_file, cloud_file) in zip(tqdm(self._data_loader.all_image_files), self._data_loader.all_cloud_files):
            projected_image = self._draw_projection_lidar_on_cam(image_file, cloud_file, transformation_mat)
            if projected_image is None:
                continue
            image_file_name = os.path.join("/tmp", "projected_" + image_file.split("/")[-1])
            cv2.imwrite(image_file_name, projected_image)

    def _draw_projection_lidar_on_cam(self, image_file: str, cloud_file: str, transformation_mat):
        image = cv2.imread(image_file)
        assert image is not None, f"failed to load {image_file}"

        rvec, _ = cv2.Rodrigues(transformation_mat[:3, :3])
        tvec = transformation_mat[:3, 3]

        plane_lidar = self._cloud_handler.run(cloud_file)
        if plane_lidar is None:
            return None

        points = plane_lidar.inliers
        projected_points, _ = cv2.projectPoints(
            points, rvec, tvec, self._image_handler.camera_info.K, self._image_handler.camera_info.dist_coeffs
        )
        projected_points = projected_points.astype(int).squeeze(axis=1)

        height, width = image.shape[:2]
        for point in projected_points:
            x, y = point
            if x < 0 or x > width - 1:
                continue
            if y < 0 or y > height - 1:
                continue
            image = cv2.circle(image, point, radius=0, color=_GREEN, thickness=10)

        return image

    @staticmethod
    def get_transformation_matrix(rpy: Array[Tuple[Literal[3]], float], tvec: Array[Tuple[Literal[3]], float]):
        transformation_mat = np.eye(4)
        rotation_mat = Rotation.from_euler("xyz", rpy).as_matrix()
        transformation_mat[:3, :3] = rotation_mat
        transformation_mat[:3, 3] = tvec

        return transformation_mat
