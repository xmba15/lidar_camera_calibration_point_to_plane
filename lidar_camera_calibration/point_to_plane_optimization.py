#!/usr/bin/env python
from typing import List, Tuple

import numpy as np
from liegroups.numpy import SE3
from scipy.optimize import least_squares

from .types import Array, Plane

try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal

__all__ = ["optimize"]


def _optimize_func(
    pose_se3, points: Array[Tuple[int, Literal[3]], float], plane_coefficients: Array[Tuple[int, Literal[4]], float]
) -> Array[Tuple[int], float]:
    transformed_points = SE3.exp(pose_se3).dot(points)
    return (np.sum(transformed_points * plane_coefficients[:, :3], axis=1) + plane_coefficients[:, 3]) * (
        plane_coefficients[:, 4] ** -1
    )


def optimize(
    all_planes_lidar: List[Plane],
    all_planes_camera: List[Plane],
):
    assert len(all_planes_lidar) == len(all_planes_camera)

    initial_transformation_mat = _direct_linear_optimize(all_planes_lidar, all_planes_camera)

    initial_pose_SE3 = SE3.from_matrix(initial_transformation_mat, normalize=True)
    initial_guess_se3 = initial_pose_SE3.log()
    plane_coefficients, points = _prepare_data(all_planes_lidar, all_planes_camera)

    res_lsq = least_squares(_optimize_func, initial_guess_se3, args=(points, plane_coefficients), loss="cauchy")

    return SE3.exp(res_lsq.x)


def _direct_linear_optimize(
    all_planes_lidar: List[Plane],
    all_planes_camera: List[Plane],
):

    Rcl = _direct_linear_optimize_rot(all_planes_lidar, all_planes_camera)
    trans = _direct_linear_optimize_trans(all_planes_lidar, all_planes_camera, Rcl)
    transformation_mat = np.eye(4)
    transformation_mat[:3, :3] = Rcl
    transformation_mat[:3, 3] = trans.squeeze()

    return transformation_mat


def _prepare_data(all_planes_lidar: List[Plane], all_planes_camera: List[Plane]):
    num_pairs = len(all_planes_lidar)
    num_points = 0
    for i in np.arange(num_pairs):
        num_points += len(all_planes_lidar[i].inliers)

    planes_camera_coefficients = np.zeros((num_points, 5))
    all_lidar_points = np.zeros((num_points, 3))

    count = 0
    for i in np.arange(num_pairs):
        cur_num_points = len(all_planes_lidar[i].inliers)
        all_lidar_points[count : count + cur_num_points, :] = all_planes_lidar[i].inliers
        planes_camera_coefficients[count : count + cur_num_points, :4] = np.tile(
            all_planes_camera[i].coefficients, cur_num_points
        ).reshape(-1, 4)
        planes_camera_coefficients[count : count + cur_num_points, 4] = np.ones(cur_num_points) * cur_num_points
        count += cur_num_points

    return planes_camera_coefficients, all_lidar_points


def _direct_linear_optimize_rot(
    all_planes_lidar: List[Plane], all_planes_camera: List[Plane]
) -> Array[Tuple[Literal[3], Literal[3]], float]:
    num_pairs = len(all_planes_lidar)
    all_planes_lidar_normals = np.zeros((num_pairs * 2, 3), dtype=np.float64)
    all_planes_camera_normals = np.zeros_like(all_planes_lidar_normals)
    for i in np.arange(num_pairs):
        all_planes_lidar_normals[i] = all_planes_lidar[i].coefficients[:3]
        all_planes_camera_normals[i] = all_planes_camera[i].coefficients[:3]

    H = all_planes_lidar_normals.T @ all_planes_camera_normals
    u, s, vh = np.linalg.svd(H)
    Rcl = vh.T @ u.T

    return Rcl


def _direct_linear_optimize_trans(
    all_planes_lidar: List[Plane],
    all_planes_camera: List[Plane],
    Rcl: Array[Tuple[Literal[3], Literal[3]], float],
):
    planes_camera_coefficients, all_lidar_points = _prepare_data(all_planes_lidar, all_planes_camera)

    A = planes_camera_coefficients[:, :3]
    rot_points = Rcl @ all_lidar_points.T
    rot_points = rot_points.T
    b = np.sum(planes_camera_coefficients[:, :3] * rot_points, axis=1) + planes_camera_coefficients[:, 3]
    b *= -1
    result = np.linalg.lstsq(A, b.reshape(-1, 1), rcond=None)
    trans = result[0]

    return trans
