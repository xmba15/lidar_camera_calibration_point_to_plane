#!/usr/bin/env python
import json
from typing import Generic, Optional, Tuple, TypeVar

import numpy as np

try:
    from typing import Annotated, Literal  # type: ignore
except ImportError:
    from typing_extensions import Annotated, Literal

__all__ = ["Array", "CameraInfo", "Plane"]


Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Array(np.ndarray, Generic[Shape, DType]):
    pass


class CameraInfo:
    def __init__(self, K: np.ndarray, dist_coeffs: np.ndarray):
        assert K[0] != 0, "invalid camera matrix"
        self._K = K.reshape(3, 3)
        self._dist_coeffs = dist_coeffs

    @property
    def K(self):
        return self._K

    @property
    def dist_coeffs(self):
        return self._dist_coeffs

    @staticmethod
    def from_json(camera_info_path: str) -> "CameraInfo":
        with open(camera_info_path, "r") as f:
            data = json.load(f)
            assert "K" in data

        try:
            dist_coeffs = np.array(data["dist_coeffs"])
        except KeyError:
            dist_coeffs = np.zeros(5, dtype=float)

        return CameraInfo(np.array(data["K"]), dist_coeffs)


class Plane:
    """
    a plane P is formulated by Ax+By+Cz+D=0.
    n=(A,B,C) is a normalized normal vector of P
    """

    def __init__(
        self, coefficients: Annotated[Tuple[float], 4], inliers: Optional[Array[Tuple[int, Literal[3]], float]] = None
    ):
        self._coefficients = coefficients
        self._inliers = inliers
        self._projections = None

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        self._coefficients = coefficients

    @property
    def inliers(self):
        return self._inliers

    @property
    def projections(self):
        if self._inliers is None:
            return None
        else:
            if self._projections is None:
                self._projections = Plane.project_points_to_plane(self._inliers, self._coefficients)
            return self._projections

    @staticmethod
    def project_points_to_plane(
        points: Array[Tuple[int, Literal[3]], float], model_coefficients: Annotated[Tuple[float], 4]
    ):
        tiled_model_coefficients = np.tile(model_coefficients, len(points)).reshape(-1, 4)
        signed_distances = np.sum(tiled_model_coefficients[:, :3] * points, axis=1) + tiled_model_coefficients[:, 3]
        return points - tiled_model_coefficients[:, :3] * signed_distances.reshape(-1, 1)

    @staticmethod
    def signed_distance_points_to_plane(
        points: Array[Tuple[int, Literal[3]], float], model_coefficients: Annotated[Tuple[float], 4]
    ):
        tiled_model_coefficients = np.tile(model_coefficients, len(points)).reshape(-1, 4)
        stacked_points = np.hstack([points, np.ones(len(points)).reshape(-1, 1)])

        return np.sum(stacked_points * tiled_model_coefficients, axis=1)

    @staticmethod
    def estimate_plane_lq(
        points: Array[Tuple[int, Literal[3]], float],
    ):
        assert len(points) > 3, "not enough number of points"
        center = np.mean(points, axis=0)
        cov = np.cov(points.T)
        u, _, _ = np.linalg.svd(cov, full_matrices=True)
        normal = u[:, 2]
        d = -normal.dot(center)
        return np.array([*normal, d])

    @staticmethod
    def from_points(points: Array[Tuple[int, Literal[3]], float]) -> "Plane":
        return Plane(Plane.estimate_plane_lq(points), points)
