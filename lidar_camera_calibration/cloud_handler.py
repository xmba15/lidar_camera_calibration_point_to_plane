#!/usr/bin/env python
import json
import logging
import os
from typing import List, Optional

import numpy as np
import open3d as o3d
import pyransac3d as pyrsc

try:
    from typing import Annotated  # type: ignore
except ImportError:
    from typing_extensions import Annotated

from .types import Plane

__all__ = ["CloudHandler"]


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s:%(message)s")
_logger = logging.getLogger(__name__)


class CloudHandler:
    def __init__(
        self,
        min_bound: Annotated[List[float], 3],
        max_bound: Annotated[List[float], 3],
        plane_ransac_thresh: float,
        plane_min_points: int,
        debug: bool = False,
    ):
        # pass through filter
        self._min_bound = min_bound
        self._max_bound = max_bound
        self._plane_ransac_thresh = plane_ransac_thresh
        self._plane_min_points = plane_min_points

        self._debug = debug

    def run(self, cloud_file: str) -> Optional[Plane]:
        pcd = o3d.t.io.read_point_cloud(cloud_file)
        assert not pcd.is_empty(), f"failed to read {cloud_file}"

        pcd.estimate_normals()
        xyz = pcd.point["positions"].numpy()
        pass_through_condition = None
        for i in np.arange(3):
            cur_dimension_condition = (xyz[:, i] >= self._min_bound[i]) & (xyz[:, i] <= self._max_bound[i])
            pass_through_condition = (
                pass_through_condition & cur_dimension_condition
                if pass_through_condition is not None
                else cur_dimension_condition
            )

        xyz = xyz[pass_through_condition]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.copy())

        # expect points on checkboard form the cluster with most points
        labels = pcd.cluster_dbscan(0.2, 3)
        values, counts = np.unique(labels, return_counts=True)
        ind = np.argmax(counts)
        pcd = pcd.select_by_index(np.where(np.array(labels) == values[ind])[0])
        xyz = np.asarray(pcd.points)

        # fit plane using ransac
        plane_fitter = pyrsc.Plane()
        plane_coeffs, inlier_indices = plane_fitter.fit(
            xyz, thresh=self._plane_ransac_thresh, minPoints=self._plane_min_points, maxIteration=1000
        )

        if len(inlier_indices) < self._plane_min_points:
            _logger.info(f"failed to extract plane of {cloud_file}")
            return None

        xyz = xyz[inlier_indices]
        pcd.points = o3d.utility.Vector3dVector(xyz.copy())

        # remove noise
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0, print_progress=False)
        xyz = np.asarray(pcd.points)

        if self._debug:
            cloud_file_name = os.path.join("/tmp", cloud_file.split("/")[-1])
            o3d.io.write_point_cloud(cloud_file_name, pcd)

        return Plane.from_points(xyz)

    @staticmethod
    def from_json(dataset_info_json: str) -> "CloudHandler":
        with open(dataset_info_json, "r") as _file:
            data = json.load(_file)
            for key in (
                "min_bound",
                "max_bound",
                "plane_ransac_thresh",
                "plane_min_points",
                "debug",
            ):
                assert key in data

        return CloudHandler(
            data["min_bound"],
            data["max_bound"],
            data["plane_ransac_thresh"],
            data["plane_min_points"],
            data["debug"],
        )
