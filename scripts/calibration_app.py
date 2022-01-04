#!/usr/bin/env python
import argparse
import logging
import os
import sys

import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
if True:
    from lidar_camera_calibration import CalibrationHandler

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s:%(message)s")
_logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser("lidar and camera calibration")
    parser.add_argument("--dataset_info_json", type=str, required=True)
    parser.add_argument("--image_list_path", type=str, required=True)
    parser.add_argument("--cloud_list_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()
    calibration_handler = CalibrationHandler(
        args.dataset_info_json, args.image_list_path, args.cloud_list_path, args.data_path
    )
    result = calibration_handler.run()
    if result is None:
        return
    rpy, tvec = result
    _logger.info(f"rpy rotation: {np.rad2deg(rpy)}[degree]")
    _logger.info(f"translation: {tvec}[meter]")


if __name__ == "__main__":
    main()
