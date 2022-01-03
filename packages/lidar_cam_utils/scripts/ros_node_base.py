#!/usr/bin/env python
import enum
import sys
from typing import List

import cv2
import message_filters
import numpy as np
import rospy

__all__ = ["RosNodeBase"]


class SynchronizerType(enum.IntEnum):
    TIME_SYNCHRONIZER = 0
    APPROX_TIME_SYNCHRONIZER = 1
    MAX = APPROX_TIME_SYNCHRONIZER


class RosNodeBase:
    def to_synchronizer(
        self,
        synchronizer_type: SynchronizerType,
        fs: List[message_filters.Subscriber],
        queue_size: int = 10,
        slop: float = 0.1,
    ) -> None:
        if synchronizer_type > SynchronizerType.MAX:
            rospy.logfatal("not supported synchronizer type")
            sys.exit()

        if synchronizer_type == 0:
            return message_filters.TimeSynchronizer(fs=fs, queue_size=queue_size)
        if synchronizer_type == 1:
            return message_filters.ApproximateTimeSynchronizer(fs=fs, queue_size=queue_size, slop=slop)

    @staticmethod
    def to_cv_image(image_msg):
        if image_msg is None:
            return None

        width = image_msg.width
        height = image_msg.height
        channels = int(len(image_msg.data) / (width * height))

        encoding = None
        if image_msg.encoding.lower() in ["rgb8", "bgr8"]:
            encoding = np.uint8
        elif image_msg.encoding.lower() == "mono8":
            encoding = np.uint8
        elif image_msg.encoding.lower() == "32fc1":
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels), dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == "mono8":
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        elif image_msg.encoding.lower() == "rgb8":
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        return cv_img
