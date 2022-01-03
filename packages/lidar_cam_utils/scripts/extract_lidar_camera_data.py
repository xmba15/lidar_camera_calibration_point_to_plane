#!/usr/bin/env python
import os
import sys

import cv2
import message_filters
import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pc2
from ros_node_base import RosNodeBase
from sensor_msgs.msg import Image, PointCloud2


class LidarCameraPairDataExtractorNode(RosNodeBase):
    def __init__(self):
        self._init_parameter()
        try:
            os.system(f"mkdir -p {self._output_path}")
        except SystemError:
            rospy.loginfo(f"failed to create {self._output_path}")
            sys.exit()

        self._subscribe()

    def _init_parameter(self):
        self._output_path = rospy.get_param("~output_path", "tmp")
        self._synchronizer_type = rospy.get_param("~synchronizer_type", 1)
        self._count = 1

    def _subscribe(self):
        self._cloud_sub = message_filters.Subscriber("~cloud", PointCloud2)
        self._camera_sub = message_filters.Subscriber("~image", Image)

        self._ts = self.to_synchronizer(
            self._synchronizer_type, fs=[self._cloud_sub, self._camera_sub], queue_size=10, slop=0.1
        )
        self._ts.registerCallback(self._callback)

    def _callback(self, cloud_msg: PointCloud2, camera_msg: Image):
        field_names = ("x", "y", "z", "intensity")

        points_data = np.array(
            [p for p in pc2.read_points(cloud_msg, skip_nans=True, field_names=field_names)],
            dtype=np.float32,
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_data[:, :3])

        pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        pcd.point["intensity"] = o3d.core.Tensor(points_data[:, 3][:, None])

        img = RosNodeBase.to_cv_image(camera_msg)
        output_path_suffix = "{:03d}".format(self._count)
        output_pc2_path = os.path.join(self._output_path, f"cloud_{output_path_suffix}.pcd")
        output_image_path = os.path.join(self._output_path, f"image_{output_path_suffix}.jpg")

        o3d.t.io.write_point_cloud(output_pc2_path, pcd)
        cv2.imwrite(output_image_path, img)

        self._count += 1


def main():
    try:
        rospy.init_node("lidar_camera_pair_data_extractor_node", anonymous=False)
        LidarCameraPairDataExtractorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logfatal("error with lidar camera extractor")
        sys.exit()


if __name__ == "__main__":
    main()
